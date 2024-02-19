namespace CodeSandbox.Leetcode.Easy;

public class MergeSortedArrays
{
    // static void Main()
    // {
    //     Merge(new []{1, 2, 3, 0, 0, 0}, 3, new [] {2, 5, 6}, 3);
    //     Merge(new []{4,0,0,0,0,0}, 1, new [] {1,2,3,5,6}, 5);
    //     Merge(new []{4,5,6,0,0,0}, 3, new [] {1,2,3}, 3);
    //
    // }
    
    static void Merge(int[] array1, int m, int[] array2, int n)
    {
        var length = m + n;
        m--;
        n--;
        for (var pointer = length - 1; pointer >=0; pointer--)
        {
            if (m < 0)
            {
                for (var i = 0; i < n + 1; i++)
                {
                    array1[i] = array2[i];
                }
                break;
            }
            if (n < 0)
                break;
            if (array1[m] > array2[n])
            {
                array1[pointer] = array1[m];
                m--;
            }
            else
            {
                array1[pointer] = array2[n];
                n--;
            }
        }
        Console.WriteLine(string.Join(" ", array1));
    }
}