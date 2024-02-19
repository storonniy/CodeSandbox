namespace CodeSandbox.Leetcode.Easy;

public class Fizzbuzz
{
    public IList<string> FizzBuzz(int n)
    {
        return Enumerable.Range(1, n)
            .Select(Meow)
            .ToArray();
    }

    private string Meow(int number)
    {
        var a = number % 3 == 0;
        var b = number % 5 == 0;
        if (a && b)
            return "FizzBuzz";
        if (a)
            return "Fizz";
        if (b)
            return "Buzz";
        return number.ToString();
    }
}