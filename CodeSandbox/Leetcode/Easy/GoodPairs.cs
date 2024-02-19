using System.Collections;

namespace CodeSandbox.Leetcode.Easy;

public class GoodPairs
{
    public int NumIdenticalPairs(int[] nums)
    {
        var dict = new Dictionary<int, int>();
        var permutationsCount = new Dictionary<int, int>();
        foreach (var t in nums)
        {
            if (!dict.ContainsKey(t))
            {
                dict.Add(t, 0);
                permutationsCount.Add(t, 0);
            }

            permutationsCount[t] += dict[t];
            dict[t]++;
        }

        return permutationsCount.Values.Sum();
    }
}