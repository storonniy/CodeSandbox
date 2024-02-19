using System.Text;

namespace CodeSandbox.Leetcode.Medium;

public class SortCharactersByFrequency
{
    public static string FrequencySort(string word)
    {
        //это ради изъеба
        var dict = new Dictionary<char, int>();
        foreach (var c in word)
        {
            if (!dict.ContainsKey(c))
                dict.Add(c, 0);
            dict[c]++;
        }

        var chars = new char[word.Length];
        var pointer = 0;
        foreach (var kp in dict.OrderByDescending(x => x.Value))
        {
            for (var i = 0; i < kp.Value; i++)
                chars[pointer++] = kp.Key;
        }

        return new string(chars);

        //а это первый вариант, где можно думать жопой и пользоваться Linq
        var sorted = word
            .GroupBy(x => x)
            .OrderByDescending(x => x.Count())
            .SelectMany(x => x.ToArray())
            .ToArray();
        return new string(sorted);
    }
}