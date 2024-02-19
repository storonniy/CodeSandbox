namespace CodeSandbox.Leetcode.Easy;

public class WordsContainingChar
{
    public IList<int> FindWordsContaining(string[] words, char x)
    {
        return words
            .Select((word, i) => (word, i))
            .Where(y => y.word.Contains(x))
            .Select(y => y.i)
            .ToArray();
    }
}