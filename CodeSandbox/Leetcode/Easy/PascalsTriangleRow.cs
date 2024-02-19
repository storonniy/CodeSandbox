namespace CodeSandbox.Leetcode.Easy;

public class PascalsTriangleRow
{
    public static IList<int> GetRow(int height)
    {
        var previous = new int[height + 1];
        previous[0] = 1;
        for (var i = 1; i < height + 1; i++)
        {
            previous[i] = 1;
            for (var k = 0; k < i; k++)
            {
                previous[k + 1] = previous[k] + previous[k + 1];
            }
            // previous[^1] = 1;
        }

        return previous;
    }
}