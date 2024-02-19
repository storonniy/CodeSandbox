namespace CodeSandbox.Leetcode.Medium;

public class ContainerWithMostWater
{

    public int MaxArea(int[] height)
    {
        var left = 0;
        var right = height.Length - 1;
        var maxArea = 0;
        while (left < right)
        {
            var dx = right - left;
            var dy = Math.Min(height[left], height[right]);
            maxArea = Math.Max(dx * dy, maxArea);
            if (height[left] < height[right])
                left++;
            else
                right--;
        }
        return maxArea;
    }
}