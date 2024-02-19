namespace CodeSandbox.Leetcode.Easy;

public class NonDivisibleSum
{
    public int DifferenceOfSums(int number, int divider)
    {
        var result = 0;
        for (var i = 1; i <= number; i++)
            result += i % divider == 0 ? -i : i;
        return result;
    }
}