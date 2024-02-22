namespace CodeSandbox.Leetcode.Medium;

public class RangeBitwiseAnd1
{
    static void Main()
    {
        RangeBitwiseAnd(5, 7);
    }
    
        public int FindJudge(int n, int[][] trust)
        {
            var trustDict = trust
                .GroupBy(x => x[0])
                .ToDictionary(x => x.Key, x => x.Select(y => y[1]).ToHashSet());
            if (trustDict.Keys.Count == n - 1)
            {
                var judge = Enumerable.Range(1, n).Sum() - trustDict.Keys.Sum();
                if (trustDict.All(x => x.Value.Contains(judge)))
                    return judge;
            }

            return -1;
        }
    
    public static int RangeBitwiseAnd(int left, int right)
    {
        var power = 0;
        while (left != right)
        {
            left >>= 1;
            right >>= 1;
            power++;
        }

        return 1 << power;
    }
}