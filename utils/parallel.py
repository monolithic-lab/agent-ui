# utils/parallel.py
import asyncio
from typing import List, Callable, Any, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def parallel_execution(
    tasks: List[Callable],
    max_workers: Optional[int] = None,
    use_threads: bool = True
) -> List[Any]:
    """
    Execute tasks in parallel
    
    Args:
        tasks: List of callable functions
        max_workers: Maximum number of workers
        use_threads: Use threads (True) or processes (False)
    
    Returns:
        List of results
    """
    if use_threads:
        executor = ThreadPoolExecutor(max_workers=max_workers)
    else:
        executor = ProcessPoolExecutor(max_workers=max_workers)
    
    try:
        futures = [executor.submit(task) for task in tasks]
        results = [future.result() for future in futures]
        return results
    finally:
        executor.shutdown(wait=True)

async def parallel_execution_async(
    coroutines: List[Callable],
    max_concurrent: Optional[int] = None
) -> List[Any]:
    """
    Execute async tasks in parallel with concurrency limit
    
    Args:
        coroutines: List of async functions
        max_concurrent: Maximum concurrent tasks
    
    Returns:
        List of results
    """
    if max_concurrent is None:
        # Run all concurrently
        tasks = [asyncio.create_task(coro()) for coro in coroutines]
        return await asyncio.gather(*tasks)
    else:
        # Limit concurrency using semaphore
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def bounded_task(coro):
            async with semaphore:
                return await coro()
        
        tasks = [asyncio.create_task(bounded_task(coro)) for coro in coroutines]
        return await asyncio.gather(*tasks)

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

async def execute_in_batches(
    items: List[Any], 
    batch_size: int,
    coro_func: Callable[[List[Any]], Any],
    max_concurrent: int = 5
) -> List[Any]:
    """
    Execute function on items in batches
    
    Args:
        items: Items to process
        batch_size: Size of each batch
        coro_func: Async function to execute on each batch
        max_concurrent: Maximum concurrent batches
    
    Returns:
        List of results
    """
    batches = chunk_list(items, batch_size)
    batch_coros = [lambda batch=b: coro_func(batch) for b in batches]
    return await parallel_execution_async(batch_coros, max_concurrent)