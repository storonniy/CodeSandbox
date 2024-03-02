using System.Text;

namespace CodeSandbox.Leetcode.Medium;

public class RangeBitwiseAnd1
{
    public int FindJudge(int n, int[][] trust)
    {
        var trustDict = trust
            .GroupBy(x => x[0])
            .ToDictionary(x => x.Key, x => x.Select(y => y[1]).ToHashSet());
        if (trustDict.Keys.Count == n - 1)
        {
            var judge = Enumerable.Range(1, n).Sum() - trustDict.Keys.Sum();
            if (trustDict.All(x => x.Value.Contains(judge)))
                return judge;
        }

        return -1;
    }

    public static bool IsStrictlyPalindromic(int n)
    {
        for (var i = 2; i <= n - 2; i++)
        {
            if (!IsPalindrome(ToBase(n, i)))
                return false;
        }

        return true;
    }

    static string ToBase(int number, int toBase)
    {
        var sb = new StringBuilder();
        while (number > 0)
        {
            sb.Append(number % toBase);
            number /= toBase;
        }

        return sb.ToString();
    }

    static bool IsPalindrome(string word)
    {
        var left = 0;
        var right = word.Length - 1;
        while (left < right)
        {
            if (word[left] != word[right])
                return false;
            left++;
            right--;
        }

        return true;
    }

    public static int RangeBitwiseAnd(int left, int right)
    {
        var power = 0;
        while (left != right)
        {
            left >>= 1;
            right >>= 1;
            power++;
        }

        return 1 << power;
    }

    public class TreeNode
    {
        public int val;
        public TreeNode left;
        public TreeNode right;

        public TreeNode(int val = 0, TreeNode left = null, TreeNode right = null)
        {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }

    public class Solution
    {

        public int DiameterOfBinaryTree(TreeNode root)
        {
            return Diameter(root).Diameter;
        }

        (int Height, int Diameter) Diameter(TreeNode root)
        {
            if (root == null)
                return (0, 0);
            var left = Diameter(root.left);
            var right = Diameter(root.right);
            var width = Math.Max(left.Diameter, right.Diameter);
            var height = 1 + Math.Max(left.Height, right.Height);
            var diameter = Math.Max(height, width);
            return (height, diameter);
        }

        int Height(TreeNode root)
        {
            if (root == null)
                return 0;
            return 1 + Math.Max(Height(root.left), Height(root.right));
        }

        public static bool IsSameTree(TreeNode root1, TreeNode root2)
        {
            if (root1 == null && root2 == null)
                return true;

            if (root1 == null || root2 == null)
                return false;

            if (root1.val != root2.val)
                return false;
            return IsSameTree(root1.left, root2.left) && IsSameTree(root1.right, root2.right);
        }

        public static int SumOfMultiples(int n)
        {
            var a = 3;
            var b = 5;
            var c = 7;
            var hashSet = new HashSet<int>();
            while (a <= n || b <= n || c <= n)
            {
                if (a <= n)
                    hashSet.Add(a);
                if (b <= n)
                    hashSet.Add(b);
                if (c <= n)
                    hashSet.Add(c);
                a += 3;
                b += 5;
                c += 7;
            }

            return hashSet.Sum();
        }

        public int[][] SortTheStudents(int[][] score, int k)
        {
            return score
                .OrderByDescending(x => x[k])
                .ToArray();
        }

        public int FindBottomLeftValue(TreeNode root)
        {
            return Dive(root).Value;
        }

        (int Depth, int Value) Dive(TreeNode root)
        {
            if (root == null)
                return (0, 0);
            if (root.left == null && root.right == null)
                return (0, root.val);
            if (root.left == null)
            {
                var r = Dive(root.right);
                return (r.Depth + 1, r.Value);
            }

            if (root.left == null)
            {
                var r = Dive(root.left);
                return (r.Depth + 1, r.Value);
            }

            var left = Dive(root.left);
            var right = Dive(root.right);
            if (left.Depth >= right.Depth)
                return (left.Depth + 1, left.Value);
            return (right.Depth + 1, right.Value);
        }

        public static string MaximumOddBinaryNumber(string line)
        {
            var sb = new StringBuilder(line);
            var pointer = 0;
            for (var i = 0; i < sb.Length; i++)
            {
                if (sb[i] == '0')
                    continue;
                sb[pointer] = '1';
                sb[i] = '0';
                pointer++;
            }

            if (pointer != 0 && pointer < line.Length)
            {
                sb[pointer - 1] = '0';
                sb[^1] = '1';
            }

            return sb.ToString();
        }
        
        static void Main()
        {
            MaximumOddBinaryNumber("010");
        }

    }
}