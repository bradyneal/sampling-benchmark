"""
File for interacting with the locally downloaded information from the OpenML
repository. This file is used to isolate openml dependencies as the only OpenML
import is in repo.py and repo.py is only imported here if the information
needed is not already accessible on the disk. Currently, the only thing this
module does is get the dataset ids for a specific task.
"""

from .io import get_downloaded_dataset_ids, is_task_file, \
                read_task_dataset_ids, write_task_dataset_ids

SUPPORTED_TASKS = ['Supervised Regression', 'Supervised Classification']


def get_dataset_ids_by_task(task):
    """
    Get the ids of all openml datasets that have the specified task type,
    reading them from a file if they've already be downloaded or downloading
    them otherwise (by a call to the repo module)
    """
    if task not in SUPPORTED_TASKS:
        raise ValueError('Unsupported task: {}\nSupported tasks: {}'
                         .format(task, SUPPORTED_TASKS))
    if is_task_file(task):
        dataset_ids_by_task = read_task_dataset_ids(task)
    else:
        # This is where the OpenML dependency will come in if the task ids
        # file isn't already on disk.
        import repo
        dataset_ids_by_task = repo.get_dataset_ids_by_task(task)
        write_task_dataset_ids(task, dataset_ids_by_task, overwrite=False)
    return dataset_ids_by_task


def get_downloaded_dataset_ids_by_task(task):
    """Get the ids of all downloaded datasets that have the specified task type"""
    downloaded_ids = set(get_downloaded_dataset_ids())
    task_ids = get_dataset_ids_by_task(task)
    downloaded_task_ids = list(downloaded_ids.intersection(task_ids))
    downloaded_task_ids.sort()
    return downloaded_task_ids
