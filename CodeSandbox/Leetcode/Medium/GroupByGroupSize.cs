namespace CodeSandbox.Leetcode.Medium;

public class GroupByGroupSize
{
    public IList<IList<int>> GroupThePeople(int[] groupSizes)
    {
        var result = new List<IList<int>>();
        var dict = new Dictionary<int, List<int>>();
        for (var i = 0; i < groupSizes.Length; i++)
        {
            var size = groupSizes[i];
            if (!dict.ContainsKey(size))
                dict.Add(size, new List<int>(size));
            dict[size].Add(i);
            if (dict[size].Count != size) continue;
            result.Add(dict[size].ToArray());
            dict[size] = new List<int>(size);
        }

        return result;
    }
}