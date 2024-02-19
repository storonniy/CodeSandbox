namespace CodeSandbox.Leetcode.Easy;

public class ReverseBitsLeetcode
{
    static uint ReverseBits(uint n)
    {
        uint result = 0;
        for (var i = 0; i < 32; i++) {
            result += (((uint)(1 << i) & n) >> i) << 31 - i;
        }
        return result;
    }
}