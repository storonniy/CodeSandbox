using System.Text;

namespace CodeSandbox.Leetcode.Easy;

public class Crappy
{
    //всякие уебищные задачки
    // wtf?? https://leetcode.com/problems/find-the-maximum-achievable-number
    public int TheMaximumAchievableX(int num, int t) => num + 2 * t;

    public double[] ConvertTemperature(double celsius)
    {
        return new[] { celsius + 273.15, celsius * 1.8 + 32.0 };
    }

    public int FinalValueAfterOperations(string[] operations)
    {
        return operations.Sum(operation => 44 - operation[1]);
    }

    static long Factorial(int n) => Enumerable.Range(1, n).Select(x => (long)x).Aggregate((a, b) => a * b);

    public string DefangIPaddr(string address)
    {
        var buffer = new StringBuilder();
        foreach (var symbol in address)
            buffer.Append(symbol == '.' ? "[.]" : symbol);
        return buffer.ToString();
    }
}