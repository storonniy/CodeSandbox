namespace CodeSandbox.Leetcode.Easy;

public class BuildArrayFromPermutation
{
    public int[] BuildArray(int[] nums) {
        for (var i = 0; i < nums.Length; i++)
        {
            (nums[nums[i]], nums[i]) = (nums[i], nums[nums[i]]);
        }

        return nums;
    }
}