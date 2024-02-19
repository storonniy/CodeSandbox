namespace CodeSandbox.Leetcode.Easy;

public class JewelsAndStones
{
    public int NumJewelsInStones(string jewels, string stones)
    {
        var jew = jewels.ToHashSet();
        return stones
            .Where(x => jew.Contains(x))
            .ToArray().Length;
    }
}