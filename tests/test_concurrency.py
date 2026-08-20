import asyncio
import unittest

from travel_planner.utils import gather_with_concurrency


class ConcurrencyTest(unittest.IsolatedAsyncioTestCase):
    async def test_hard_concurrency_limit(self):
        active = 0
        peak = 0
        lock = asyncio.Lock()

        async def worker(value):
            nonlocal active, peak
            async with lock:
                active += 1
                peak = max(peak, active)
            await asyncio.sleep(0.01)
            async with lock:
                active -= 1
            return value

        factories = [lambda value=value: worker(value) for value in range(9)]
        results = await gather_with_concurrency(3, factories)
        self.assertEqual(list(range(9)), results)
        self.assertEqual(3, peak)


if __name__ == "__main__":
    unittest.main()
