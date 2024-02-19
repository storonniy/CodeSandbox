namespace CodeSandbox.Leetcode.Medium;

public class PrefixXor
{
    public int[] FindArray(int[] pref)
    {
        for (var i = pref.Length - 1; i > 1; i--)
        {
            pref[i] = pref[i - 1] ^ pref[i];
        }

        return pref;
    }
}