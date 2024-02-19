namespace CodeSandbox.Leetcode.Medium;

public class ArraysWithMaxDifference
{
    //https://leetcode.com/problems/divide-array-into-arrays-with-max-difference/?envType=daily-question&envId=2024-02-01
    public int[][] DivideArray(int[] nums, int k)
    {
        Array.Sort(nums);
        var result = new int[nums.Length / 3][];
        for (var i = 0; i < nums.Length; i += 3)
        {
            if (nums[i + 2] - nums[i] > k)
                return Array.Empty<int[]>() ;
            result[i / 3] = new[] { nums[i], nums[i + 1], nums[i + 2] };
        }

        return result;
    }
}