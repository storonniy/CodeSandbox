namespace CodeSandbox.Leetcode.Medium;

public class FirstPalindromicString
{
    public string FirstPalindrome(string[] words)
    {
        foreach (var word in words)
        {
            if (IsPalindrome(word))
                return word;
        }

        return string.Empty;
    }

    bool IsPalindrome(string word)
    {
        var left = 0;
        var right = word.Length - 1;
        while (left <= right)
        {
            if (word[left] != word[right])
                return false;
            left++;
            right--;
        }

        return true;
    }
}