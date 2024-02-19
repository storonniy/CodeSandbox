using System.Text;

namespace CodeSandbox.Leetcode.Medium;

public class StringToInteger
{

    public int MyAtoi(string line)
    {
        var i = 0;
        var sb = new StringBuilder();
        while (i < line.Length && line[i] == ' ')
            i++;
        if (i < line.Length && new HashSet<char> {'+', '-'}.Contains(line[i]))
        {
            sb.Append(line[i]);
            i++;
        }

        if (i < line.Length && sb.Length > 0 && new HashSet<char>  { '+', '-' }.Contains(line[i]))
            return 0;
        while (i < line.Length && line[i] >=  '0' && line[i] <= '9')
        {
            sb.Append(line[i]);
            i++;
        }

        if (sb.Length == 0)
            return 0;
        if (new HashSet<char>  {'+', '-'}.Contains(sb[0]) && sb.Length == 1)
            return 0;
        var result = sb.ToString();
        if (!long.TryParse(result, out var bigInteger))
            return result[0] == '-' ? int.MinValue : int.MaxValue;
        if (bigInteger > int.MaxValue)
            return int.MaxValue;
        if (bigInteger < int.MinValue)
            return int.MinValue;
        return (int)bigInteger;
    }
}