namespace CodeSandbox.Leetcode.Easy;

public class PascalsTriangle
{
    private static void Print(IList<IList<int>> sequence)
    {
        foreach (var row in sequence)
        {
            Console.WriteLine(string.Join(' ', row));
        }
    }
    
    public static IList<IList<int>> Generate(int height)
    {
        var result = new List<IList<int>> { new[] {1}};
        for (var i = 1; i < height; i++)
        {
            var previous = result[i - 1];
            var row = new int[i + 1];
            row[0] = 1;
            row[^1] = 1;
            for (var k = 0; k < previous.Count - 1; k++)
                row[k + 1] = previous[k] + previous[k + 1];
            result.Add(row);
        }
        return result;
    }
}