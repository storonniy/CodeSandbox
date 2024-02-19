using System.Collections;

namespace CodeSandbox.Leetcode.Medium;

public class FurthestReachedBuilding
{
    //https://leetcode.com/problems/furthest-building-you-can-reach/description/?envType=daily-question&envId=2024-02-17
    //17.02.24
    
    public static int FurthestBuilding(int[] heights, int bricks, int ladders)
    {
        var queue = new PriorityQueue<int, int>();
        for (var i = 0; i < heights.Length - 1; i++)
        {
            var diff = heights[i + 1] - heights[i];
            if (diff <= 0)
                continue;
            bricks -= diff;
            queue.Enqueue(diff, -diff);
            if (bricks < 0)
            {
                if (ladders == 0)
                    return i;
                bricks += queue.Dequeue();
                ladders--;
            }
        }
        return heights.Length - 1;
    }
}