import os
from datetime import datetime


class FileUtils(object):

    @staticmethod
    def ensure_dir(dir_path: str) -> None:

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    @staticmethod
    def create_dated_directory(parent_directory: str) -> str:

        today = datetime.today()
        dir_path = os.path.join(parent_directory, today.strftime('%Y%m%d'))
        FileUtils.ensure_dir(dir_path)
        return dir_path

    @staticmethod
    def create_timed_directory(
            parent_directory: str, suffix: str = None
    ) -> str:

        today = datetime.now()
        dir_name = (
            f"{today.strftime('%Y%m%d%H%M')}_{suffix}"
            if suffix else today.strftime('%Y%m%d%H%M')
        )
        dir_path = os.path.join(parent_directory, dir_name)
        FileUtils.ensure_dir(dir_path)
        return dir_path