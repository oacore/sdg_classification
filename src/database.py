import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import mysql.connector
from mysql.connector import Error
from mysql.connector.pooling import MySQLConnectionPool

logger = logging.getLogger(__name__)

DbConfig = dict[str, str | int | None]
DbRecord = dict[str, Any]
RelatedCollections = dict[str, dict[int, list[DbRecord]]]


class DatabaseConnectionPool:
    """
    manages connection pool.
    DO NOT CREATE INSTANCES DIRECTLY, should be used ONLY by AppContext.
    """

    def __init__(self, db_config: DbConfig, pool_size: int) -> None:
        try:
            self._pool = MySQLConnectionPool(
                pool_name="chars_indexer_pool",
                pool_size=pool_size,
                pool_reset_session=True,
                charset="utf8mb4",
                collation="utf8mb4_unicode_ci",
                autocommit=False,
                **db_config,
            )
        except Error as e:
            raise RuntimeError(f"Failed to create MySQL connection pool: {e}") from e

    @contextmanager
    def connection(self) -> Iterator[Any]:
        """
        Yield a connection from the pool and guarantee its return.
        Rolls back on exception, commits on clean exit.
        """
        conn = self._pool.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()  # returns to pool, does not destroy

    def close(self) -> None:
        if hasattr(self, "_pool"):
            del self._pool


class DatabaseConnection:
    """
    Pure query executor.
    Accepts a borrowed connection from the pool — does not own or manage its lifecycle.
    """

    RELATED_COLLECTION_CHUNK_SIZE = 10_000
    RELATED_ID_QUERY_CHUNK_SIZE = 1_000

    def __init__(self, connection: mysql.connector.MySQLConnection, entity_type: str) -> None:
        self._conn = connection
        self._entity_type = entity_type

    def _fetch_related_collection(
            self,
            table_name: str,
            fk_column: str,
            entity_ids: list[int],
    ) -> dict[int, list[DbRecord]]:
        """
        Fetch related rows for the exact entity IDs in bounded IN queries.
        Results from every ID chunk are combined into one entity-keyed mapping.
        """
        if not entity_ids:
            return {}

        unique_entity_ids = list(dict.fromkeys(entity_ids))
        logger.info(
            "Fetching related rows for %s in %s-ID chunks (total IDs: %s)",
            table_name,
            self.RELATED_ID_QUERY_CHUNK_SIZE,
            len(unique_entity_ids),
        )

        by_output: dict[int, list[DbRecord]] = {}
        cursor = self._conn.cursor(dictionary=True)
        try:
            for start in range(
                    0,
                    len(unique_entity_ids),
                    self.RELATED_ID_QUERY_CHUNK_SIZE,
            ):
                chunk_ids = unique_entity_ids[
                    start:start + self.RELATED_ID_QUERY_CHUNK_SIZE
                ]
                placeholders = ", ".join(["%s"] * len(chunk_ids))
                query = f"""
                    SELECT *
                    FROM {table_name}
                    WHERE {fk_column} IN ({placeholders})
                """
                cursor.execute(query, tuple(chunk_ids))
                for row in cursor.fetchall():
                    entity_id = row.get(fk_column)
                    if entity_id is None:
                        continue
                    by_output.setdefault(entity_id, []).append(row)
        except mysql.connector.Error as err:
            logger.error(
                f"Failed to fetch related rows from {table_name}.{fk_column}: {err}"
            )
            raise
        finally:
            cursor.close()

        return by_output

    def _fetch_related_collection_joined(
            self,
            table_name: str,
            join_table: str,
            join_column_main: str,
            join_column_second: str,
            cte_columns: list[str],
            added_column: str,
            added_column_alias: str,
            fk_column: str,
            output_ids: tuple[int, int],
    ) -> dict[int, list[DbRecord]]:
        if not output_ids:
            return {}

        selected_columns = ", ".join(f"main_filtered.{column}" for column in cte_columns)
        query = f"""
            WITH main_filtered as(
            SELECT *
            FROM {table_name}
            WHERE {fk_column} BETWEEN %s AND %s)
            SELECT {selected_columns}, {join_table}.{added_column} as {added_column_alias}
            FROM main_filtered
            JOIN {join_table} ON main_filtered.{join_column_main} = {join_table}.{join_column_second}
        """

        cursor = self._conn.cursor(dictionary=True)
        try:
            cursor.execute(query, (output_ids[0], output_ids[-1]))
            rows = cursor.fetchall()
        except mysql.connector.Error as err:
            logger.error(
                f"Failed to fetch joined related rows from {table_name}.{fk_column}: {err}"
            )
            raise
        finally:
            cursor.close()

        by_output: dict[int, list[DbRecord]] = {}
        for row in rows:
            oid = row.get(fk_column)
            if oid is None:
                continue
            by_output.setdefault(oid, []).append(row)
        return by_output

    def _fetch_work_data_providers(
            self,
            work_ids: list[int],
    ) -> dict[int, list[DbRecord]]:
        if not work_ids:
            return {}

        unique_work_ids = list(dict.fromkeys(work_ids))
        by_work: dict[int, list[DbRecord]] = {}
        cursor = self._conn.cursor(dictionary=True)
        try:
            for start in range(
                    0,
                    len(unique_work_ids),
                    self.RELATED_ID_QUERY_CHUNK_SIZE,
            ):
                chunk_ids = unique_work_ids[
                    start:start + self.RELATED_ID_QUERY_CHUNK_SIZE
                ]
                placeholders = ", ".join(["%s"] * len(chunk_ids))
                query = f"""
                    SELECT
                        wdp.work_id,
                        wdp.provider_id,
                        dp.name
                    FROM work_data_providers wdp
                    JOIN data_provider dp ON wdp.provider_id = dp.id
                    WHERE wdp.work_id IN ({placeholders})
                """
                cursor.execute(query, tuple(chunk_ids))
                for row in cursor.fetchall():
                    work_id = row.get("work_id")
                    if work_id is None:
                        continue
                    by_work.setdefault(work_id, []).append({
                        "id": row.get("provider_id"),
                        "name": row.get("name"),
                    })
        except mysql.connector.Error:
            raise
        finally:
            cursor.close()

        return by_work

    def _fetch_related_collection_by_ids(
            self,
            table_name: str,
            fk_column: str,
            entity_ids: list[int],
    ) -> dict[int, list[DbRecord]]:
        return self._fetch_related_collection(table_name, fk_column, entity_ids)

    def query_main_table(self) -> str:
        if self._entity_type == "output":
            query = """SELECT o.id,
                       o.title,
                       o.abstract,
                       o.doi,
                       o.oai,
                       o.download_url,
                       o.fulltext_status,
                       o.year_published,
                       o.accepted_date,
                       o.created_date,
                       o.deposited_date,
                       o.published_date,
                       o.updated_date,
                       o.last_update,
                       o.language,
                       o.publisher,
                       o.repository_document,
                       o.dataprovider_id AS dataprovider_id,
                       dp.name AS dataprovider_name,
                       o.license,
                       o.deleted,
                       o.disabled,
                       o.created_at,
                       o.updated_at
	                FROM output o
	                LEFT JOIN data_provider dp ON o.dataprovider_id = dp.id
	                WHERE o.id BETWEEN %s AND %s AND o.deleted = 'ALLOWED'
	                ORDER BY o.id
	            """
        elif self._entity_type == "work":
            query = """SELECT w.id,
	                       w.title,
	                       w.abstract,
                       w.document_type,
                       w.download_url,
                       (
                           SELECT wo.output_id
                           FROM work_outputs wo
                           JOIN output o
                             ON o.id = wo.output_id
                           WHERE wo.work_id = w.id
                             AND o.fulltext_status = 'AVAILABLE'
                           ORDER BY wo.output_id
                           LIMIT 1
                       ) AS available_output_id,
                       w.field_of_study,
                       w.citation_count,
                       w.year_published,
                       w.accepted_date,
                       w.created_date,
                       w.deposited_date,
                       w.published_date,
                       w.updated_date,
                       w.language,
                       w.publisher,
                       w.pubmed_id,
                       w.created_at,
	                       w.updated_at
	                FROM work w
	                WHERE w.id BETWEEN %s AND %s
	                ORDER BY w.id
	                        """
        else:
            raise ValueError(f"Unsupported entity type: {self._entity_type}")
        return query

    def query_related_tables(self, result_ids: list[int]) -> RelatedCollections:
        logger.info(f"Querying related tables for {self._entity_type}")
        if self._entity_type == "output":
            result = {
                "authors":      self._fetch_related_collection("output_authors",      "output_id", result_ids),
                "contributors": self._fetch_related_collection("output_contributors", "output_id", result_ids),
                "versions":     self._fetch_related_collection("output_versions",     "output_id", result_ids),
                "identifiers":  self._fetch_related_collection("output_identifiers",  "output_id", result_ids),
                "references":   self._fetch_related_collection("output_references",   "output_id", result_ids),
                "links":        self._fetch_related_collection("output_links",        "output_id", result_ids),
                "tags":         self._fetch_related_collection("output_tags",         "output_id", result_ids),
                "setSpecs":     self._fetch_related_collection("output_set_specs",    "output_id", result_ids),
                "subjects":     self._fetch_related_collection("output_subjects",     "output_id", result_ids),
                "journals":     self._fetch_related_collection("output_journals",     "output_id", result_ids),
                "urls":         self._fetch_related_collection("output_urls",         "output_id", result_ids),
                "sdg":          self._fetch_related_collection("output_sdg",          "output_id", result_ids),
            }
        elif self._entity_type == "work":
            result = {
                "authors":          self._fetch_related_collection("work_authors",      "work_id", result_ids),
                "data_providers":   self._fetch_work_data_providers(result_ids),
                "contributors":     self._fetch_related_collection("work_contributors", "work_id", result_ids),
                "identifiers":      self._fetch_related_collection("work_identifiers",  "work_id", result_ids),
                "journals":         self._fetch_related_collection("work_journals",     "work_id", result_ids),
                "links":            self._fetch_related_collection("work_links",        "work_id", result_ids),
                "references":       self._fetch_related_collection("work_references",   "work_id", result_ids),
            }
        else:
            raise ValueError(f"Unsupported entity type: {self._entity_type}")
        return result

    def query_related_tables_for_ids(self, entity_ids: list[int]) -> RelatedCollections:
        logger.info(f"Querying related tables for {self._entity_type} exact IDs")
        if self._entity_type == "output":
            result = {
                "authors":      self._fetch_related_collection_by_ids("output_authors",      "output_id", entity_ids),
                "contributors": self._fetch_related_collection_by_ids("output_contributors", "output_id", entity_ids),
                "versions":     self._fetch_related_collection_by_ids("output_versions",     "output_id", entity_ids),
                "identifiers":  self._fetch_related_collection_by_ids("output_identifiers",  "output_id", entity_ids),
                "references":   self._fetch_related_collection_by_ids("output_references",   "output_id", entity_ids),
                "links":        self._fetch_related_collection_by_ids("output_links",        "output_id", entity_ids),
                "tags":         self._fetch_related_collection_by_ids("output_tags",         "output_id", entity_ids),
                "setSpecs":     self._fetch_related_collection_by_ids("output_set_specs",    "output_id", entity_ids),
                "subjects":     self._fetch_related_collection_by_ids("output_subjects",     "output_id", entity_ids),
                "journals":     self._fetch_related_collection_by_ids("output_journals",     "output_id", entity_ids),
                "urls":         self._fetch_related_collection_by_ids("output_urls",         "output_id", entity_ids),
                "sdg":          self._fetch_related_collection_by_ids("output_sdg",          "output_id", entity_ids),
            }
        elif self._entity_type == "work":
            result = {
                "authors":          self._fetch_related_collection_by_ids("work_authors",      "work_id", entity_ids),
                "data_providers":   self._fetch_work_data_providers(entity_ids),
                "contributors":     self._fetch_related_collection_by_ids("work_contributors", "work_id", entity_ids),
                "identifiers":      self._fetch_related_collection_by_ids("work_identifiers",  "work_id", entity_ids),
                "journals":         self._fetch_related_collection_by_ids("work_journals",     "work_id", entity_ids),
                "links":            self._fetch_related_collection_by_ids("work_links",        "work_id", entity_ids),
                "references":       self._fetch_related_collection_by_ids("work_references",   "work_id", entity_ids),
            }
        else:
            raise ValueError(f"Unsupported entity type: {self._entity_type}")
        return result

    def _attach_related_collections(
            self,
            records: list[DbRecord],
            chunk_size: int = RELATED_COLLECTION_CHUNK_SIZE,
    ) -> None:
        for start in range(0, len(records), chunk_size):
            chunk = records[start:start + chunk_size]
            chunk_ids = [
                int(row["id"])
                for row in chunk
                if row.get("id") is not None
            ]
            if not chunk_ids:
                continue

            related = self.query_related_tables_for_ids(chunk_ids)

            for row in chunk:
                oid = row.get("id")
                if oid is None:
                    continue
                for key, by_output in related.items():
                    row[key] = by_output.get(oid, [])

    def fetch_records(self, from_id: int, to_id: int) -> list[DbRecord]:
        cursor = self._conn.cursor(dictionary=True)
        try:
            logger.debug(f"Fetching {self._entity_type} records for {from_id} to {to_id}")
            query = self.query_main_table()
            cursor.execute(query, (from_id, to_id))

            result = cursor.fetchall()
            logger.info(
                f"Fetched {self._entity_type} records for {from_id} to {to_id}, "
                f"there are {len(result)} records"
            )
        except mysql.connector.Error as err:
            logger.error(f"Failed to fetch {self._entity_type} records for {from_id} to {to_id}: {err}")
            raise
        finally:
            cursor.close()

        if not result:
            return []

        entity_ids = [
            int(row["id"])
            for row in result
            if row.get("id") is not None
        ]
        if not entity_ids:
            return result
        related = self.query_related_tables(entity_ids)

        for row in result:
            oid = row.get("id")
            if oid is None:
                continue
            for key, by_output in related.items():
                row[key] = by_output.get(oid, [])

        return result

    def get_current_timestamp(self) -> datetime:
        cursor = self._conn.cursor(dictionary=True)
        try:
            cursor.execute("SELECT NOW() AS db_now")
            row = cursor.fetchone() or {}
        finally:
            cursor.close()

        current_time = row.get("db_now")
        if not isinstance(current_time, datetime):
            raise RuntimeError(
                f"Database returned invalid current timestamp: {current_time!r}"
            )
        return current_time

    def get_updates_bucket(self) -> DbRecord | None:
        table_name = f"{self._entity_type}_updates_buckets"
        cursor = self._conn.cursor(dictionary=True)
        try:
            cursor.execute(f"""
                SELECT bucket_id, id_from, id_to
                FROM {table_name}
                WHERE status = 'pending'
                ORDER BY bucket_id ASC
                LIMIT 1
                FOR UPDATE
            """)
            bucket = cursor.fetchone()
            if bucket is None:
                return None

            cursor.execute(f"""
                UPDATE {table_name}
                SET status = 'in_progress'
                WHERE bucket_id = %s
                  AND status = 'pending'
            """, (bucket["bucket_id"],))
            if cursor.rowcount != 1:
                raise RuntimeError(
                    f"Failed to claim updates bucket {bucket['bucket_id']}"
                )
            return bucket
        except mysql.connector.Error as err:
            logger.error(
                f"Failed to claim updates bucket for {self._entity_type}: {err}"
            )
            raise
        finally:
            cursor.close()

    def mark_updates_bucket_completed(self, bucket_id: int) -> None:
        self._set_updates_bucket_status(bucket_id, "completed")

    def mark_updates_bucket_failed(self, bucket_id: int) -> None:
        self._set_updates_bucket_status(bucket_id, "failed")

    def _set_updates_bucket_status(self, bucket_id: int, status: str) -> None:
        if status not in {"completed", "failed"}:
            raise ValueError(f"Unsupported updates bucket status: {status}")

        cursor = self._conn.cursor()
        try:
            cursor.execute(f"""
                UPDATE {self._entity_type}_updates_buckets
                SET status = %s
                WHERE bucket_id = %s
                  AND status = 'in_progress'
            """, (status, bucket_id))
        except mysql.connector.Error as err:
            logger.error(
                f"Failed to mark updates bucket {bucket_id} as {status}: {err}"
            )
            raise
        finally:
            cursor.close()

    def fetch_updated_records_batch(
            self,
            id_from: int,
            id_to: int,
    ) -> list[DbRecord]:
        if id_from > id_to:
            raise ValueError("id_from must be less than or equal to id_to")

        if self._entity_type == "output":
            query = """
                SELECT o.id,
                       o.title,
                       o.abstract,
                       o.doi,
                       o.oai,
                       o.download_url,
                       o.fulltext_status,
                       o.year_published,
                       o.accepted_date,
                       o.created_date,
                       o.deposited_date,
                       o.published_date,
                       o.updated_date,
                       o.last_update,
                       o.language,
                       o.publisher,
                       o.repository_document,
                       o.dataprovider_id AS dataprovider_id,
                       dp.name AS dataprovider_name,
                       o.license,
                       o.deleted,
                       o.disabled,
                       o.created_at,
                       o.updated_at
                FROM output o
                JOIN output_entity_create_status ecs
                  ON o.id = ecs.id_document
                LEFT JOIN data_provider dp ON o.dataprovider_id = dp.id
                WHERE ecs.ai_search_update = 1
                  AND o.id BETWEEN %s AND %s
                  AND o.deleted = 'ALLOWED'
                ORDER BY o.id
            """
        elif self._entity_type == "work":
            query = """
                SELECT w.id,
                       w.title,
                       w.abstract,
                       w.document_type,
                       w.download_url,
                       (
                           SELECT wo.output_id
                           FROM work_outputs wo
                           JOIN output o
                             ON o.id = wo.output_id
                           WHERE wo.work_id = w.id
                             AND o.fulltext_status = 'AVAILABLE'
                           ORDER BY wo.output_id
                           LIMIT 1
                       ) AS available_output_id,
                       w.field_of_study,
                       w.citation_count,
                       w.year_published,
                       w.accepted_date,
                       w.created_date,
                       w.deposited_date,
                       w.published_date,
                       w.updated_date,
                       w.language,
                       w.publisher,
                       w.pubmed_id,
                       w.created_at,
                       w.updated_at
                FROM work w
                JOIN work_entity_create_status ecs
                  ON w.id = ecs.work_id
                WHERE ecs.ai_search_update = 1
                  AND w.id BETWEEN %s AND %s
                ORDER BY w.id
            """
        else:
            raise ValueError(f"Unsupported entity type: {self._entity_type}")

        cursor = self._conn.cursor(dictionary=True)
        try:
            cursor.execute(
                query,
                (id_from, id_to),
            )
            result = cursor.fetchall()
            logger.info(
                f"Fetched {len(result)} updated {self._entity_type} records "
                f"between ids {id_from} and {id_to}"
            )
        except mysql.connector.Error as err:
            logger.error(
                f"Failed to fetch updated records for {self._entity_type}: {err}"
            )
            raise
        finally:
            cursor.close()

        self._attach_related_collections(result)
        return result

    def create_indexing_statuses(
            self,
            status_records: list[tuple[int, int, str, str]],
            created_at: datetime,
    ) -> int:
        if not status_records:
            return 0

        cursor = self._conn.cursor()
        try:
            cursor.executemany(
                """
                INSERT INTO indexing_status (
                    id_start,
                    id_end,
                    entity,
                    shard_type,
                    blob_name,
                    status,
                    created_at
                )
                VALUES (%s, %s, %s, %s, %s, 'Created', %s)
                """,
                [
                    (
                        id_start,
                        id_end,
                        self._entity_type,
                        shard_type,
                        blob_name,
                        created_at,
                    )
                    for id_start, id_end, shard_type, blob_name in status_records
                ],
            )
            return cursor.rowcount
        finally:
            cursor.close()

    def mark_ai_search_documents_current(self, entity_ids: list[int]) -> int:
        unique_ids = list(dict.fromkeys(entity_ids))
        if not unique_ids:
            return 0

        if self._entity_type == "work":
            table_name = "work_entity_create_status"
            id_column = "work_id"
        elif self._entity_type == "output":
            table_name = "output_entity_create_status"
            id_column = "id_document"
        else:
            raise ValueError(f"Unsupported entity type: {self._entity_type}")

        affected_rows = 0
        cursor = self._conn.cursor()
        try:
            for start in range(0, len(unique_ids), self.RELATED_ID_QUERY_CHUNK_SIZE):
                chunk_ids = unique_ids[
                    start:start + self.RELATED_ID_QUERY_CHUNK_SIZE
                ]
                placeholders = ", ".join(["%s"] * len(chunk_ids))
                cursor.execute(
                    f"""
                    UPDATE {table_name}
                    SET ai_search_update = 0
                    WHERE {id_column} IN ({placeholders})
                    """,
                    tuple(chunk_ids),
                )
                affected_rows += cursor.rowcount
        except mysql.connector.Error as err:
            logger.error(
                "Failed to clear AI Search update flags for %s: %s",
                self._entity_type,
                err,
            )
            raise
        finally:
            cursor.close()

        logger.info(
            "Cleared AI Search update flags for %s: requested=%s, changed=%s",
            self._entity_type,
            len(unique_ids),
            affected_rows,
        )
        return affected_rows

    def fetch_outputs_by_dataprovider_chunk(
            self,
            dataprovider_id: int,
            last_id: int = 0,
            chunk_size: int = 1000,
    ) -> list[DbRecord]:
        """
        Fetch a chunk of output records (id, title, abstract) for a single dataprovider_id
        using key-set pagination (id > last_id).
        """
        query = """
            SELECT id, title, abstract
            FROM output
            WHERE dataprovider_id = %s AND id > %s AND deleted = 'ALLOWED'
            ORDER BY id ASC
            LIMIT %s
        """
        cursor = self._conn.cursor(dictionary=True)
        try:
            cursor.execute(query, (dataprovider_id, last_id, chunk_size))
            return cursor.fetchall()
        except mysql.connector.Error as err:
            logger.error(
                "Failed to fetch output records for dataprovider_id %s (last_id %s): %s",
                dataprovider_id,
                last_id,
                err,
            )
            raise
        finally:
            cursor.close()

    def save_article_sdg_classifications(
            self,
            records: list[tuple[int, str, float]],
    ) -> int:
        """
        Batch insert or update classification results into article_sdg_classification.
        records: list of tuples (id_document, sdg_class, confidence_score)
        """
        if not records:
            return 0

        query = """
            INSERT INTO article_sdg_classification (id_document, sdg_class, confidence_score)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE
                sdg_class = VALUES(sdg_class),
                confidence_score = VALUES(confidence_score)
        """
        cursor = self._conn.cursor()
        try:
            cursor.executemany(query, records)
            return cursor.rowcount
        except mysql.connector.Error as err:
            logger.error("Failed to save article_sdg_classification records: %s", err)
            raise
        finally:
            cursor.close()


class BucketOrchestrator:
    """
    Manages all interactions with the {entity_type}_buckets table.
    Uses a dedicated stable connection — not from the pool.
    Owned exclusively by AppContext.
    Thread-safe: all operations are protected by a single lock so that
    parallel pipeline threads can call mark_complete / mark_failed concurrently.
    """

    MAX_RETRIES = 3

    def __init__(self, db_config: DbConfig, entity_type: str, column_prefix: str = "") -> None:
        if column_prefix and not column_prefix.replace("_", "").isalnum():
            raise ValueError(f"Invalid bucket column prefix: {column_prefix}")

        self._conn = mysql.connector.connect(
            **db_config,
            autocommit=False,
            charset="utf8mb4",
            collation="utf8mb4_unicode_ci",
        )
        self._cursor = self._conn.cursor(dictionary=True)
        self._lock = threading.Lock()
        self._table = f"{entity_type}_buckets"
        self._column_prefix = column_prefix
        self._status_column = f"{column_prefix}status"
        self._error_message_column = f"{column_prefix}error_message"
        self._retry_count_column = f"{column_prefix}retry_count"
        self._started_at_column = f"{column_prefix}started_at"
        self._completed_at_column = f"{column_prefix}completed_at"

    def _ensure_connected(self) -> None:
        try:
            self._conn.ping(reconnect=False)
            return
        except mysql.connector.Error as err:
            logger.warning(
                "Bucket orchestrator connection is unavailable; reconnecting: %s",
                err,
            )

        try:
            self._cursor.close()
        except Exception:
            pass

        self._conn.ping(reconnect=True, attempts=3, delay=1)
        self._cursor = self._conn.cursor(dictionary=True)

    @contextmanager
    def _connection_guard(self) -> Iterator[None]:
        with self._lock:
            self._ensure_connected()
            yield

    def _current_db_time(self) -> Any:
        self._cursor.execute("SELECT NOW() AS db_now")
        row = self._cursor.fetchone()
        return row["db_now"]

    def get_earliest_unprocessed_message_id(self) -> int | None:
        with self._connection_guard():
            self._cursor.execute(f"""
                SELECT MIN(message_id) AS earliest
                FROM {self._table}
                WHERE message_id NOT IN (
                    SELECT DISTINCT message_id
                    FROM {self._table}
                    WHERE {self._started_at_column} IS NOT NULL
                )
            """)
            row = self._cursor.fetchone()
            return row["earliest"] if row else None

    def get_next_pending_message(self) -> int | None:
        with self._connection_guard():
            self._cursor.execute(f"""
                SELECT MIN(message_id) as next_message
                FROM {self._table}
                WHERE {self._status_column} = 'pending'
            """)
            row = self._cursor.fetchone()
            return row["next_message"] if row else None

    def claim_message(self, message_id: int) -> list[DbRecord]:
        with self._connection_guard():
            claim_started_at = self._current_db_time()
            self._cursor.execute(f"""
                UPDATE {self._table}
                SET {self._status_column} = 'in_progress',
                    {self._started_at_column} = %s
                WHERE message_id = %s
                  AND {self._status_column} = 'pending'
            """, (claim_started_at, message_id))
            claimed_count = self._cursor.rowcount
            self._conn.commit()

            if claimed_count == 0:
                return []

            self._cursor.execute(f"""
                SELECT bucket_id, id_from, id_to
                FROM {self._table}
                WHERE message_id = %s
                  AND {self._status_column} = 'in_progress'
                  AND {self._started_at_column} = %s
            """, (message_id, claim_started_at))
            return self._cursor.fetchall()

    def collect_failed_buckets(
            self,
            limit: int = 10,
            exclude_bucket_ids: set[int] | None = None,
    ) -> list[DbRecord]:
        with self._connection_guard():
            retry_order_column = self._started_at_column
            excluded_ids = sorted(exclude_bucket_ids or set())
            exclude_clause = ""
            select_params: list[Any] = [self.MAX_RETRIES]
            if excluded_ids:
                excluded_placeholders = ",".join(["%s"] * len(excluded_ids))
                exclude_clause = f"AND bucket_id NOT IN ({excluded_placeholders})"
                select_params.extend(excluded_ids)
            select_params.append(limit)

            self._cursor.execute(f"""
                SELECT bucket_id, id_from, id_to, {self._retry_count_column} AS retry_count
                FROM {self._table}
                WHERE {self._status_column} = 'failed'
                AND {self._retry_count_column} < %s
                {exclude_clause}
                ORDER BY {retry_order_column} ASC
                LIMIT %s
            """, tuple(select_params))
            buckets = self._cursor.fetchall()

            if buckets:
                claim_started_at = self._current_db_time()
                ids = [b["bucket_id"] for b in buckets]
                placeholders = ",".join(["%s"] * len(ids))
                update_params: list[Any] = [claim_started_at]
                update_params.extend(ids)
                update_params.append(self.MAX_RETRIES)

                self._cursor.execute(f"""
                    UPDATE {self._table}
                    SET {self._status_column} = 'in_progress',
                        {self._started_at_column} = %s
                    WHERE bucket_id IN ({placeholders})
                      AND {self._status_column} = 'failed'
                      AND {self._retry_count_column} < %s
                """, tuple(update_params))
                claimed_count = self._cursor.rowcount
                self._conn.commit()

                if claimed_count == 0:
                    return []

                self._cursor.execute(f"""
                    SELECT bucket_id, id_from, id_to, {self._retry_count_column} AS retry_count
                    FROM {self._table}
                    WHERE bucket_id IN ({placeholders})
                      AND {self._status_column} = 'in_progress'
                      AND {self._started_at_column} = %s
                """, (*ids, claim_started_at))
                return self._cursor.fetchall()

            return []

    def mark_complete(self, bucket_id: int) -> None:
        with self._connection_guard():
            self._cursor.execute(f"""
                UPDATE {self._table}
                SET {self._status_column} = 'completed',
                    {self._completed_at_column} = NOW(),
                    {self._error_message_column} = NULL
                WHERE bucket_id = %s
            """, (bucket_id,))
            self._conn.commit()

    def mark_failed(self, bucket_id: int, error: str) -> None:
        with self._connection_guard():
            self._cursor.execute(f"""
                UPDATE {self._table}
                SET {self._status_column} = 'failed',
                    {self._error_message_column} = %s,
                    {self._retry_count_column} = {self._retry_count_column} + 1
                WHERE bucket_id = %s
            """, (error[:1000], bucket_id))
            self._conn.commit()

    def mark_message_complete(self, message_id: int) -> None:
        with self._connection_guard():
            self._cursor.execute(f"""
                UPDATE {self._table}
                SET {self._status_column} = 'completed',
                    {self._completed_at_column} = NOW()
                WHERE message_id = %s
                AND {self._status_column} = 'in_progress'
            """, (message_id,))
            self._conn.commit()

    def release_stale(self, timeout_minutes: int = 30) -> int:
        with self._connection_guard():
            self._cursor.execute(f"""
                UPDATE {self._table}
                SET {self._status_column} = 'pending',
                    {self._started_at_column} = NULL
                WHERE {self._status_column} = 'in_progress'
                AND {self._started_at_column} < NOW() - INTERVAL %s MINUTE
            """, (timeout_minutes,))
            self._conn.commit()
            return self._cursor.rowcount

    def get_progress(self) -> dict[str, dict[str, int | None]]:
        with self._connection_guard():
            self._cursor.execute(f"""
                SELECT
                    {self._status_column} AS status,
                    COUNT(*) as count,
                    SUM(record_count) as total_records
                FROM {self._table}
                GROUP BY {self._status_column}
            """)
            rows = self._cursor.fetchall()
            progress: dict[str, dict[str, int | None]] = {}
            for row in rows:
                status_value: Any = row.get("status")
                if not isinstance(status_value, str):
                    raise RuntimeError(
                        f"Database returned invalid bucket status: {status_value!r}"
                    )

                count_value: Any = row.get("count")
                total_records_value: Any = row.get("total_records")
                progress[status_value] = {
                    "buckets": (
                        int(count_value) if count_value is not None else None
                    ),
                    "records": (
                        int(total_records_value)
                        if total_records_value is not None
                        else None
                    ),
                }
            return progress

    def close(self) -> None:
        try:
            self._cursor.close()
            self._conn.close()
        except Exception:
            pass
