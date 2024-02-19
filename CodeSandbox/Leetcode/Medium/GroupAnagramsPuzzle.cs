namespace CodeSandbox.Leetcode.Medium;

public class GroupAnagramsPuzzle
{
    static IList<IList<string>> GroupAnagrams(string[] words)
    {
        var dict = new Dictionary<string, List<string>>();
        foreach (var word in words)
        {
            var chars = word.ToCharArray();
            Array.Sort(chars);
            var hash = new string(chars);
            if (!dict.ContainsKey(hash))
                dict.Add(hash, new List<string>());
            dict[hash].Add(word);
        }

        return new List<IList<string>>(dict.Values);
    }
}