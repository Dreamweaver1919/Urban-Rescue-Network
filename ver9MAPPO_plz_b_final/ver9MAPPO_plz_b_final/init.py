
import random
import Task as TaskModule
import Rescuer as RescuerModule

def generate_random_tasks(num_tasks, grid_size, min_victims=10, max_victims=30, max_deadline=200):
    tasks = []
    for i in range(num_tasks):
        task = TaskModule.Task(
            task_id=i,
            grid_size=grid_size,
            victim=random.randint(min_victims, max_victims),
            deadline=random.randint(50, max_deadline)
        )
        tasks.append(task)
    return tasks

def init_rescuers(num_rescuers, grid_size):
    return [RescuerModule.Rescuer(i, grid_size) for i in range(num_rescuers)]