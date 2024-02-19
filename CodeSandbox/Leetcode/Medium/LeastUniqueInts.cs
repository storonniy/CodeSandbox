namespace CodeSandbox.Leetcode.Medium;

public class LeastUniqueInts
{
    public int FindLeastNumOfUniqueInts(int[] nums, int k)
    {
        var dict = nums
            .GroupBy(x => x)
            .ToDictionary(x => x.Key, x => x.Count());
        foreach (var kp in dict.OrderBy(x => x.Value))
        {
            if (kp.Value > k)
                break;
            dict.Remove(kp.Key);
            k -= kp.Value;
        }

        return dict.Keys.Count;
    }
}