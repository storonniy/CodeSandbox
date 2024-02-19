using System.Text;

namespace CodeSandbox.Leetcode.Medium;

public class SequentalDigits
{
    public static IList<int> SequentialDigits(int low, int high)
    {
        var result = new List<int>();
        var length = low.ToString().Length;
        var start = low.ToString()[0] - '0';
        var growing = low;
        while (growing <= high)
        {
            if (start + length - 1 > 9)
            {
                length++;
                start = 1;
            }

            var str = string.Join("", Enumerable.Range(start, length).ToArray());
            if (str.Length > high.ToString().Length)
                return result;
            growing = int.Parse(str);
            if (growing < low)
            {
                start++;
                continue;
            }
            if (growing <= high)
            {
                result.Add(growing);
                start++;
            }
        }

        return result;
    }
}