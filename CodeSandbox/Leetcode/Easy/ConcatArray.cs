namespace CodeSandbox.Leetcode.Easy;

public class ConcatArray
{
    public int[] GetConcatenation(int[] nums)
    {
        var result = new int[nums.Length];
        for (var i = 0; i < 2 * nums.Length; i++)
        {
            result[i] = nums[i];
            result[nums.Length + i] = nums[i];
        }
        return result;
    }
}