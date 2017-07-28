"""
File for interacting with openml metadata (e.g. getting dataset ids that
correspond to a particular task such as "Supervised Regression").
"""

import openml
import pandas as pd

from .io import get_downloaded_dataset_ids

SUPPORTED_TASKS = ['Supervised Regression', 'Supervised Classification']


def get_dataset_ids():
    """Get the ids of all openml datasets that don't have missing values"""
    dataset_metadata = openml.datasets.list_datasets()
    metadata_df = pd.DataFrame.from_dict(dataset_metadata, orient='index')
    filtered_df = metadata_df[metadata_df.NumberOfInstancesWithMissingValues == 0]
    return filtered_df.did.values


def get_dataset_ids_by_task(task):
    """Get the ids of all openml datasets that have the specified task type"""
    if task not in SUPPORTED_TASKS:
        raise ValueError('Unsupported task: {}\nSupported tasks: {}'
                         .format(task, SUPPORTED_TASKS))
    tasks = openml.tasks.list_tasks()
    tasks_df = pd.DataFrame.from_dict(tasks, orient='index')
    return tasks_df[tasks_df.task_type == task].did.values


def get_downloaded_dataset_ids_by_task(task):
    """Get the ids of all downloaded datasets that have the specified task type"""
    downloaded_ids = set(get_downloaded_dataset_ids())
    task_ids = get_dataset_ids_by_task(task)
    return downloaded_ids.intersection(task_ids)
