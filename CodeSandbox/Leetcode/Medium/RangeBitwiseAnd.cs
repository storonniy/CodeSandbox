using System.Collections;
using System.Diagnostics.Metrics;
using System.Drawing;
using System.Reflection.Metadata;
using System.Text;
using System.Text.RegularExpressions;
using System.Xml.Linq;
using Microsoft.VisualBasic;
using Newtonsoft.Json;

namespace CodeSandbox.Leetcode.Medium;

public class FindElements
{
    public FindElements(TreeNode root)
    {
        tree = root;
        root.val = 0;
        Go(root.left, 1);
        Go(root.right, 2);
    }

    private TreeNode tree;
    private HashSet<int> buffer = new() { 0 };

    private void Go(TreeNode node, int newValue)
    {
        if (node == null)
            return;
        buffer.Add(newValue);
        node.val = newValue;
        Go(node.left, 2 * newValue + 1);
        Go(node.right, 2 * newValue + 2);
    }

    public bool Find(int target)
    {
        return buffer.Contains(target);
    }
}

public class NumberContainers
{
    public NumberContainers()
    {
    }

    public class ProductOfNumbers
    {
        private int[] buffer = new int[100000];
        int index = 0;

        private void StartAgain()
        {
            buffer = new int[100000];
            index = 0;
        }

        public ProductOfNumbers()
        {
        }

        public void Add(int num)
        {
            if (num == 0)
                StartAgain();
            else
            {
                if (index == 0)
                {
                    buffer[index] = num;
                    return;
                }

                buffer[index + 1] = buffer[index] * num;
                index++;
            }
        }

        public int GetProduct(int k)
        {
            if (k > buffer.Length)
                return 0;
            return buffer[index] / buffer[index - k];
        }
    }

    private Dictionary<int, int> dict = new();
    private Dictionary<int, SortedSet<int>> valueToIndexMap = new();

    public void Change(int index, int number)
    {
        if (!dict.TryGetValue(index, out var n))
            dict[index] = number;
        else
        {
            valueToIndexMap[n].Remove(index);
        }

        if (!valueToIndexMap.ContainsKey(number))
        {
            valueToIndexMap.Add(number, new SortedSet<int>());
        }

        valueToIndexMap[number].Add(index);
    }

    public int Find(int number)
    {
        return valueToIndexMap.ContainsKey(number) && valueToIndexMap[number].Count > 0
            ? valueToIndexMap[number].First()
            : -1;
    }
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

public class GenericType<T>
    where T : class
{
}

class AnotherGenericType<T> : GenericType<T>
    where T : class
{
}

public class RangeBitwiseAnd1
{
    public static string Simplify(string source)
    {
        return Regex.Replace(source, @"\s+", " ")
            .Trim()
            .ToLower()
            .Replace('ё', 'e')
            .Replace('й', 'и');
    }

    public int CountGoodTriplets(int[] arr, int a, int b, int c)
    {
        var result = 0;
        for (var i = 0; i < arr.Length; i++)
        {
            for (var j = i + 1; j < arr.Length; j++)
            {
                var ij = Math.Abs(arr[i] - arr[j]);
                if (ij > a)
                    continue;
                for (var k = j + 1; k < arr.Length; k++)
                {
                    var jk = Math.Abs(arr[j] - arr[k]);
                    var ik = Math.Abs(arr[i] - arr[k]);
                    if (jk <= b && ik <= c)
                        result++;
                }
            }
        }

        return result;
    }

    public static int CountGoodNumbers(long n)
    {
        if (n == 1)
            return 5;
        var basic = 1000000007;
        var count = 1 + (int)((n & 1) << 2);
        var x = 20;
        for (n >>= 1;; x = (int)((long)x * x % basic))
        {
            if ((n & 1) > 0)
                count = (int)((long)count * x % basic);
            if ((n >>= 1) == 0)
                return count;
        }
    }

    public static long CountGoodIntegers(int n, int k)
    {
        long count = 0;
        var start = (long)Math.Pow(10, (n - 1) / 2);
        var skip = n % 2;
        var unique = new HashSet<string>();
        for (var x = start; x < 10 * start; x++)
        {
            var value = x.ToString();
            value += new string(value.ToCharArray().Reverse().ToArray(), skip, value.Length - skip);
            var palindrome = long.Parse(value);
            if (palindrome % k != 0)
                continue;
            var chars = value.ToCharArray();
            Array.Sort(chars);
            unique.Add(new string(chars));
        }

        var factorial = new long[n + 1];
        factorial[0] = 1;
        for (var i = 1; i <= n; i++)
            factorial[i] = factorial[i - 1] * i;
        foreach (var x in unique)
        {
            var c = new int[10];
            foreach (var d in x)
                c[d - '0']++;
            var arrangements = (n - c[0]) * factorial[n - 1];
            foreach (var i in c)
            {
                arrangements /= factorial[i]; // Divide by factorial of counts
            }

            count += arrangements;
        }

        return count;
    }

    static bool CanBePalindrome(long x, int n)
    {
        var values = new int[10];
        while (x > 0)
        {
            values[x % 10]++;
            x /= 10;
        }

        if (n % 2 == 0 && values.All(i => i % 2 == 0))
            return true;
        if (n % 2 != 0)
        {
            var hasOdd = false;
            var hasZeroAsEven = false;
            var evenCount = 0;
            for (var i = 0; i < values.Length; i++)
            {
                if (values[i] == 0)
                    continue;
                if (values[i] % 2 == 1)
                {
                    if (hasOdd)
                        return false;
                    hasOdd = true;
                }
                else
                {
                    evenCount++;
                    if (i == 0)
                        hasZeroAsEven = true;
                }
            }

            if (evenCount == 1 && hasZeroAsEven)
                return false;
        }

        return true;
    }

    public int CountSymmetricIntegers(int low, int high)
    {
        var count = 0;
        for (var i = low; i <= high; i++)
            if (IsSymmetric(i.ToString()))
                count++;
        return count;
    }

    bool IsSymmetric(string value)
    {
        if (value.Length % 2 == 1)
            return false;
        var left = 0;
        var right = 0;
        for (var i = 0; i < value.Length / 2; i++)
            left += value[i] - '0';
        for (var i = value.Length / 2; i < value.Length; i++)
            right += value[i] - '0';
        return left == right;
    }

    public long NumberOfPowerfulInt(long start, long finish, int limit, string s)
    {
        var startStr = (start - 1).ToString();
        var finishStr = finish.ToString();
        return Calculate(finishStr, s, limit) - Calculate(startStr, s, limit);
    }

    private long Calculate(string x, string s, int limit)
    {
        if (x.Length < s.Length)
            return 0;

        if (x.Length == s.Length)
            return string.CompareOrdinal(x, s) >= 0 ? 1 : 0;

        if (s.Any(c => c - '0' > limit))
            return 0;

        long count = 0;
        var prefixLen = x.Length - s.Length;
        var tight = true;

        for (var i = 0; i < prefixLen; i++)
        {
            var maxDigit = tight ? x[i] - '0' : limit;

            for (var d = 0; d < maxDigit && d <= limit; d++)
            {
                count += (long)Math.Pow(limit + 1, prefixLen - 1 - i);
            }

            if (x[i] - '0' > limit)
                return count;

            if (x[i] - '0' < maxDigit)
                tight = false;
        }

        var candidate = x[..prefixLen] + s;
        if (string.CompareOrdinal(candidate, x) <= 0)
            count++;
        return count;
    }

    public static int MinOperations(int[] nums, int k)
    {
        var distinct = new HashSet<int>(nums.Length);
        foreach (var x in nums)
        {
            if (x > k)
                distinct.Add(x);
            else if (x < k)
                return -1;
        }

        return distinct.Count;
    }

    public static int MinimumOperations(int[] nums)
    {
        var dict = new Dictionary<int, int>();
        foreach (var x in nums)
        {
            dict.TryAdd(x, 0);
            dict[x]++;
        }

        if (dict.Keys.Count == nums.Length)
            return 0;
        var pointer = 0;
        var count = 0;
        while (pointer < nums.Length && dict.Keys.Count < nums.Length - pointer)
        {
            count++;
            for (var i = 0; i < 3 && pointer + i < nums.Length; i++)
            {
                dict[nums[pointer + i]]--;
                if (dict[nums[pointer + i]] == 0)
                    dict.Remove(nums[pointer + i]);
            }

            pointer += 3;
        }

        return count;
    }

    public static bool CanPartition(int[] nums)
    {
        var sum = nums.Sum();
        if (sum % 2 == 1)
            return false;
        var target = sum / 2;
        var dp = new bool[target + 1];
        dp[0] = true;
        foreach (var x in nums)
        {
            if (x > target)
                return false;
            if (dp[target - x])
                return true;
            for (var j = target; j >= x; j--)
            {
                dp[j] |= dp[j - x];
            }
        }

        return false;
    }

    static bool BacktrackSum(int position, List<int> nums, int sum, int target)
    {
        if (position == nums.Count)
            return false;
        foreach (var x in nums)
        {
            if (sum + x > target)
                continue;
            if (sum + x == target)
                return true;
            sum += x;
            nums.RemoveAt(position);
            var solution = BacktrackSum(position + 1, nums, sum, target);
            if (solution)
                return true;
            sum -= x;
        }

        return sum == target;
    }

    public static IList<int> LargestDivisibleSubset(int[] nums)
    {
        if (nums.Length == 1)
            return nums;
        Array.Sort(nums);
        var n = nums.Length;
        var dp = new int[n];
        var prevIndex = new int[n];
        var maxIndex = 0;

        for (var i = 0; i < n; i++)
        {
            dp[i] = 1;
            prevIndex[i] = -1;
            for (var j = i - 1; j >= 0; j--)
            {
                if (nums[i] % nums[j] != 0 || dp[i] >= dp[j] + 1)
                    continue;
                dp[i] = dp[j] + 1;
                prevIndex[i] = j;
            }

            if (dp[i] > dp[maxIndex])
                maxIndex = i;
        }

        var subset = new List<int>();
        while (maxIndex >= 0)
        {
            subset.Add(nums[maxIndex]);
            maxIndex = prevIndex[maxIndex];
        }

        return subset;
    }

    private TreeNode result = null;
    private int maxDepth;

    public TreeNode LcaDeepestLeaves(TreeNode root)
    {
        GetDepth(root, 0);
        return result;
    }

    public int GetDepth(TreeNode node, int level)
    {
        if (node == null)
            return 0;

        int left = GetDepth(node.left, level + 1);
        int right = GetDepth(node.right, level + 1);

        if (left == right && level + left >= maxDepth)
        {
            maxDepth = level + left;
            result = node;
        }

        return Math.Max(left, right) + 1;
    }

    void Go(TreeNode root, int level)
    {
        if (root == null)
            return;
        if (root.left == null && root.right == null)
        {
        }
    }

    public static long MaximumTripletValue(int[] nums)
    {
        var differences = new int[nums.Length];
        var maxLeft = int.MinValue;
        var rights = new int[nums.Length];
        var maxRight = int.MinValue;
        for (var i = nums.Length - 1; i > 0; i--)
        {
            maxRight = Math.Max(maxRight, nums[i]);
            rights[i] = maxRight;
        }

        var max = long.MinValue;

        for (var i = 1; i < nums.Length - 1; i++)
        {
            maxLeft = Math.Max(nums[i - 1], maxLeft);
            differences[i] = maxLeft - nums[i];
            max = Math.Max((long)differences[i] * rights[i + 1], max);
        }

        return max < 0 ? 0 : max;
    }

    public long MostPoints(int[][] questions)
    {
        var points = new int[questions.Length];
        for (var i = questions.Length - 1; i >= 0; i--)
        {
            var next = i + questions[i][1] + 1;
            if (next < questions.Length)
                points[i] = points[next] + questions[i][0];
            else
                points[i] = questions[i][0];
            if (i == questions.Length - 1)
                continue;
            points[i] = Math.Max(points[i], points[i + 1]);
        }

        return points[0];
    }

    public static IList<int> PartitionLabels(string s)
    {
        var lastIndexes = new Dictionary<char, int>();
        for (var i = s.Length - 1; i >= 0; i--)
        {
            if (!lastIndexes.ContainsKey(s[i]))
                lastIndexes.Add(s[i], i);
        }

        var result = new List<int>();
        var length = 1;
        var end = 0;
        for (var i = 0; i < s.Length; i++)
        {
            var currentEnd = lastIndexes[s[i]];
            end = Math.Max(end, currentEnd);
            if (i == end)
            {
                result.Add(length);
                length = 1;
                continue;
            }

            length++;
        }

        return result;
    }

    public static int[] MaxPoints(int[][] grid, int[] queries)
    {
        var values = new Dictionary<int, int>();
        foreach (var x in queries)
            values.TryAdd(x, 0);
        var sorted = values.Keys.ToArray();
        Array.Sort(sorted);
        var height = grid.Length;
        var width = grid[0].Length;
        var visited = new HashSet<(int X, int Y)>();
        var queue = new PriorityQueue<(int X, int Y), int>();
        for (var i = 0; i < sorted.Length; i++)
        {
            var query = sorted[i];
            if (i == 0)
                queue.Enqueue((0, 0), grid[0][0]);
            while (queue.Count > 0)
            {
                var (xi, yi) = queue.Peek();
                if (query <= grid[yi][xi])
                    break;
                var (x, y) = queue.Dequeue();
                if (!visited.Add((x, y)))
                    continue;
                var adjacent = new List<(int X, int Y)>();
                if (x - 1 >= 0)
                    adjacent.Add((x - 1, y));
                if (x + 1 < width)
                    adjacent.Add((x + 1, y));
                if (y - 1 >= 0)
                    adjacent.Add((x, y - 1));
                if (y + 1 < height)
                    adjacent.Add((x, y + 1));
                foreach (var p in adjacent)
                {
                    queue.Enqueue((p.X, p.Y), grid[p.Y][p.X]);
                }
            }

            values[query] = visited.Count;
        }

        for (var i = 0; i < queries.Length; i++)
            queries[i] = values[queries[i]];
        return queries;
    }

    public int MaximumScore(IList<int> nums, int k)
    {
        var n = nums.Count;
        var numArr = new int[n];
        var maxNum = int.MinValue;
        for (var i = 0; i < n; i++)
        {
            numArr[i] = nums[i];
            maxNum = Math.Max(maxNum, nums[i]);
        }

        nums.CopyTo(numArr, 0);
        var primeScores = PrimeScoresSieve(maxNum);

        var left = new int[n];
        var right = new int[n];
        var stack = new Stack<int>();

        for (var i = 0; i < n; i++)
        {
            while (stack.Count > 0 && primeScores[numArr[stack.Peek()]] < primeScores[numArr[i]])
            {
                stack.Pop();
            }

            left[i] = stack.Count == 0 ? i + 1 : i - stack.Peek();
            stack.Push(i);
        }

        stack.Clear();
        for (var i = n - 1; i >= 0; i--)
        {
            while (stack.Count > 0 && primeScores[numArr[stack.Peek()]] <= primeScores[numArr[i]])
            {
                stack.Pop();
            }

            right[i] = stack.Count == 0 ? n - i : stack.Peek() - i;
            stack.Push(i);
        }

        var freq = new long[n];
        for (var i = 0; i < n; i++)
        {
            freq[i] = (long)left[i] * right[i];
        }

        Array.Sort(numArr, freq, Comparer<int>.Create((a, b) => b.CompareTo(a)));
        var currentIndex = 0;
        var maximumScore = 1;
        while (k > 0)
        {
            maximumScore = (int)((long)maximumScore *
                                 ModPow(numArr[currentIndex], Math.Min(freq[currentIndex], k), 1000000007) %
                                 1000000007);
            k -= (int)Math.Min(freq[currentIndex], k);
            currentIndex++;
        }

        return maximumScore;
    }

    int[] PrimeScoresSieve(int num)
    {
        var primeScores = new int[num + 1];

        for (var i = 2; i <= num; i++)
        {
            if (primeScores[i] != 0) continue;
            for (var j = i; j <= num; j += i)
            {
                primeScores[j]++;
            }
        }

        return primeScores;
    }

    int ModPow(long value, long exponent, int modulus)
    {
        var result = 1L;
        value %= modulus;
        while (exponent > 0)
        {
            if ((exponent & 1) == 1)
                result = result * value % modulus;
            value = value * value % modulus;
            exponent >>= 1;
        }

        return (int)result;
    }

    static (HashSet<(int X, int Y)> visited, HashSet<(int X, int Y)> nonVisited) Bfs(int[][] grid,
        (int X, int Y) start,
        int value)
    {
        var height = grid.Length;
        var width = grid[0].Length;
        var visited = new HashSet<(int X, int Y)>();
        var nonVisited = new HashSet<(int X, int Y)>();
        var queue = new PriorityQueue<(int X, int Y), int>();
        queue.Enqueue(start, grid[start.X][start.Y]);
        while (queue.Count > 0)
        {
            var (x, y) = queue.Dequeue();
            if (!visited.Add((x, y)))
                continue;
            var adjacent = new List<(int X, int Y)>();
            if (x - 1 >= 0)
                adjacent.Add((x - 1, y));
            if (x + 1 < width)
                adjacent.Add((x + 1, y));
            if (y - 1 >= 0)
                adjacent.Add((x, y - 1));
            if (y + 1 < height)
                adjacent.Add((x, y + 1));
            foreach (var p in adjacent)
            {
                if (value <= grid[p.Y][p.X])
                    nonVisited.Add((p.X, p.Y));
                else if (!visited.Contains(p))
                    queue.Enqueue((p.X, p.Y), grid[p.Y][p.X]);
            }
        }

        return (visited, nonVisited);
    }

    public static int MinimumIndex(IList<int> nums)
    {
        var frequency = new Dictionary<int, int>();
        foreach (var x in nums)
        {
            frequency.TryAdd(x, 0);
            frequency[x]++;
        }

        var dominant = int.MinValue;
        var max = int.MinValue;
        foreach (var x in frequency)
        {
            if (x.Value > max)
            {
                max = x.Value;
                dominant = x.Key;
            }
        }

        var f = 0;
        var generalFrequency = frequency[dominant];
        var n = nums.Count;
        for (var i = 0; i < nums.Count; i++)
        {
            if (nums[i] == dominant)
                f++;
            if (f > (i + 1) / 2 && generalFrequency - f > (n - i - 1) / 2)
                return i;
        }

        return -1;
    }

    public int MinOperations(int[][] grid, int x)
    {
        var dict = new Dictionary<int, int>();
        var values = grid.SelectMany(v => v).ToArray();
        foreach (var v in grid.SelectMany(v => v))
        {
            dict.TryAdd(v, 0);
            dict[v]++;
        }


        var remainder = values.First() % x;
        if (dict.Keys.Any(v => v % x != remainder))
            return -1;
        Array.Sort(values);
        var median = values[values.Length / 2];
        var count = 0;
        foreach (var v in dict.Keys)
        {
            count += dict[v] * Math.Abs(median - v) / x;
        }

        return count;
    }

    public static bool CheckValidCuts(int n, int[][] rectangles)
    {
        var merged = rectangles
            .Select(x => new[] { x[0], x[2] })
            .OrderBy(i => i[0])
            .Aggregate(new List<int[]>(), (res, cur) =>
            {
                if (res.Count == 0 || res[^1][1] <= cur[0])
                    res.Add(cur);
                else
                    res[^1][1] = Math.Max(res[^1][1], cur[1]);
                return res;
            })
            .ToArray();
        if (merged.Length > 2)
            return true;
        merged = rectangles
            .Select(x => new[] { x[1], x[3] })
            .OrderBy(i => i[0])
            .Aggregate(new List<int[]>(), (res, cur) =>
            {
                if (res.Count == 0 || res[^1][1] <= cur[0])
                    res.Add(cur);
                else
                    res[^1][1] = Math.Max(res[^1][1], cur[1]);
                return res;
            })
            .ToArray();
        if (merged.Length > 2)
            return true;
        return false;
    }

    public static int CountDays(int days, int[][] meetings)
    {
        var merged = meetings
            .OrderBy(i => i[0])
            .Aggregate(new List<int[]>(), (res, cur) =>
            {
                if (res.Count == 0 || res[^1][1] < cur[0])
                    res.Add(cur);
                else
                    res[^1][1] = Math.Max(res[^1][1], cur[1]);
                return res;
            })
            .Select(x => new { Start = x[0], End = x[1] })
            .ToArray();
        var hours = merged[0].Start - 1 + days - merged.Last().End;
        for (var i = 0; i < merged.Length - 1; i++)
        {
            hours += merged[i + 1].Start - merged[i].End - 1;
        }

        return hours;
    }

    public int CountPaths(int n, int[][] roads)
    {
        var mod = 1_000_000_007;
        var neighbours = new List<(int, int)>[n];
        for (var i = 0; i < n; i++)
            neighbours[i] = new List<(int, int)>();

        foreach (var road in roads)
        {
            neighbours[road[0]].Add((road[1], road[2]));
            neighbours[road[1]].Add((road[0], road[2]));
        }

        var shortestTime = new long[n];
        Array.Fill(shortestTime, long.MaxValue);
        var dp = new int[n];
        var queue = new SortedSet<(long, int)> { (0, 0) };

        shortestTime[0] = 0;
        dp[0] = 1;

        while (queue.Count > 0)
        {
            var (time, node) = queue.Min;
            queue.Remove(queue.Min);
            if (time > shortestTime[node]) continue;
            foreach (var (x, t) in neighbours[node])
            {
                if (time + t < shortestTime[x])
                {
                    queue.Remove((shortestTime[x], x));
                    shortestTime[x] = time + t;
                    dp[x] = dp[node];
                    queue.Add((shortestTime[x], x));
                }
                else if (time + t == shortestTime[x])
                    dp[x] = (dp[x] + dp[node]) % mod;
            }
        }

        return dp[n - 1];
    }

    public static int CountCompleteComponents(int n, int[][] edges)
    {
        var neighbours = new Dictionary<int, List<int>>();
        foreach (var e in edges)
        {
            if (!neighbours.ContainsKey(e[0]))
                neighbours.Add(e[0], new List<int>());
            if (!neighbours.ContainsKey(e[1]))
                neighbours.Add(e[1], new List<int>());
            neighbours[e[0]].Add(e[1]);
            neighbours[e[1]].Add(e[0]);
        }

        var visited = new HashSet<int>();
        var count = n - neighbours.Keys.Count;
        foreach (var x in neighbours.Keys)
        {
            if (visited.Contains(x))
                continue;
            var connectivity = Bfs(neighbours, x);
            foreach (var c in connectivity)
                visited.Add(c);
            var weight = connectivity.Count;
            var isComplete = true;
            foreach (var c in connectivity)
            {
                if (neighbours[c].Count != weight - 1)
                {
                    isComplete = false;
                    break;
                }
            }

            if (isComplete)
                count++;
        }

        return count;
    }

    static HashSet<int> Bfs(Dictionary<int, List<int>> neighbours, int x)
    {
        var visited = new HashSet<int>();
        var queue = new Queue<int>();
        queue.Enqueue(x);
        while (queue.Count > 0)
        {
            var y = queue.Dequeue();
            if (visited.Contains(y))
                break;
            visited.Add(y);
            foreach (var node in neighbours[y])
            {
                if (!queue.Contains(node) && !visited.Contains(node))
                    queue.Enqueue(node);
            }
        }

        return visited;
    }

    public IList<string> FindAllRecipes(string[] recipes, IList<IList<string>> ingredients, string[] supplies)
    {
        var parts = new HashSet<string>(supplies);
        var recipesDict = new Dictionary<string, IList<string>>();
        for (var i = 0; i < recipes.Length; i++)
            recipesDict.Add(recipes[i], ingredients[i]);
        var result = new List<string>();
        var queue = new Queue<string>(recipes);
        var lastSize = -1;
        while (parts.Count > lastSize)
        {
            lastSize = parts.Count;
            var queueSize = queue.Count;
            while (queueSize-- > 0)
            {
                var r = queue.Dequeue();
                var part = recipesDict[r];
                var found = true;
                if (part.Any(x => !parts.Contains(x)))
                {
                    found = false;
                    queue.Enqueue(r);
                }

                if (!found)
                    continue;
                parts.Add(r);
                result.Add(r);
            }
        }

        return result;
    }

    public string MaximumTime(string time)
    {
        var sb = new StringBuilder(time);
        if (sb[0] == '?' && sb[1] == '?')
        {
            sb[0] = '2';
            sb[1] = '3';
        }
        else if (sb[0] == '?')
        {
            if (sb[1] < '4')
                sb[0] = '2';
            else
                sb[0] = '1';
        }
        else
        {
            if (sb[0] == '2')
                sb[1] = '3';
            else
                sb[1] = '9';
        }

        if (sb[3] == '?')
            sb[3] = '5';
        if (sb[4] == '?')
            sb[4] = '9';
        return sb.ToString();
    }

    public bool CheckOnesSegment(string s)
    {
        var p = 0;
        while (p < s.Length && s[p] == '1')
        {
            p++;
        }

        while (p < s.Length && s[p] == '0')
        {
            p++;
        }

        return p == s.Length;
    }

    public int[] MinimumCost(int n, int[][] edges, int[][] query)
    {
        var parent = new int[n][];
        var result = new int[query.Length];
        for (var i = 0; i < n; i++)
            parent[i] = new int[2] { i, int.MaxValue };
        foreach (var e in edges)
        {
            var p1 = GetParent(parent, e[0]);
            var p2 = GetParent(parent, e[1]);
            parent[p2[0]][0] = p1[0];
            p1[1] = p1[1] & p2[1] & e[2];
        }

        for (var i = 0; i < query.Length; i++)
        {
            var p1 = GetParent(parent, edges[i][0]);
            var p2 = GetParent(parent, edges[i][1]);
            if (p1[0] == p2[0])
            {
                result[i] = p1[1];
            }
            else
                result[i] = -1;
        }

        return result;
    }

    private int[] GetParent(int[][] parent, int node)
    {
        if (parent[node][0] == node)
            return parent[node];
        var p = GetParent(parent, parent[node][0]);
        parent[node][0] = p[0];
        return p;
    }

    public static int MinOperations(int[] nums)
    {
        var min = 0;
        var left = 0;
        while (left < nums.Length && nums[left] == 1)
            left++;
        var step = 0;
        while (left < nums.Length - 2)
        {
            step++;
            for (var j = 0; j < 3; j++)
                nums[left + j] = Math.Abs(nums[left + j] - 1);
            while (left < nums.Length && nums[left] == 1)
                left++;
        }

        return left == nums.Length ? step : -1;
    }

    public static int LongestNiceSubarray(int[] nums)
    {
        var maxLength = 1;
        var allBits = nums[0];
        var length = 1;
        var left = 0;
        var right = 1;
        while (left < nums.Length)
        {
            while (right < nums.Length && (allBits & nums[right]) == 0)
            {
                allBits |= nums[right];
                maxLength = Math.Max(maxLength, right - left + 1);
                right++;
            }

            allBits &= ~nums[left];
            left++;
        }

        return maxLength;
    }

    public bool DivideArray(int[] nums)
    {
        var dict = new Dictionary<int, int>();
        foreach (var x in nums)
        {
            dict.TryAdd(x, 0);
            dict[x]++;
        }

        foreach (var x in dict.Values)
        {
            if (x % 2 == 1)
                return false;
        }

        return true;
    }

    public long RepairCars(int[] ranks, int cars)
    {
        long left = ranks.Min();
        var right = left * cars * cars;
        while (left < right)
        {
            var time = left + (right - left) / 2;
            long repairedCars = 0;
            foreach (var r in ranks)
                repairedCars += (int)Math.Sqrt(time / r);
            if (repairedCars >= cars)
                right = time;
            else
                left = time + 1;
        }

        return left;
    }

    public int MinCapability(int[] nums, int k)
    {
        var left = nums.Min();
        var right = nums.Max();
        while (left < right)
        {
            var capability = (right + left + 1) / 2;
            var count1 = 0;
            for (var i = 0; i < nums.Length; i += 2)
            {
                if (nums[i] <= capability)
                    count1++;
            }

            var count2 = 0;
            for (var i = 1; i < nums.Length; i += 2)
            {
                if (nums[i] <= capability)
                    count2++;
            }

            var count = Math.Max(count1, count2);
            if (count >= k)
                left = capability;
            else
                right = capability - 1;
        }

        return left;
    }

    public static int MaximumCandies(int[] candies, long k)
    {
        var max = 0;
        long sum = 0;
        foreach (var x in candies)
        {
            max = Math.Max(max, x);
            sum += x;
        }

        if (candies.Sum() < k)
            return 0;
        long left = 1;
        long right = max;
        while (left < right)
        {
            long candiesCount = (right + left + 1) / 2;
            long piles = 0;
            for (var i = 0; i < candies.Length; i++)
                piles += candies[i] / candiesCount;
            if (piles >= k)
                left = candiesCount;
            else
                right = candiesCount - 1;
        }

        return (int)right;
    }

    public int MinZeroArray(int[] nums, int[][] queries)
    {
        var left = 0;
        var right = nums.Length;
        if (!IsZero(nums, queries, right))
            return -1;
        while (left < right)
        {
            var m = left + (right - left) / 2;
            if (IsZero(nums, queries, m))
                right = m;
            else left = m + 1;
        }

        return left;
    }

    bool IsZero(int[] nums, int[][] queries, int k)
    {
        var difference = new int[nums.Length];
        for (var i = 0; i < k; i++)
        {
            var l = queries[i][0];
            var r = queries[i][1];
            var value = queries[i][2];
            difference[l] += value;
            if (r + 1 < nums.Length)
                difference[r + 1] -= value;
        }

        var sum = 0;
        for (var i = 0; i < nums.Length; i++)
        {
            sum += difference[i];
            if (sum < nums[i])
                return false;
        }

        return true;
    }

    public static int MaximumCount(int[] nums)
    {
        var lastNegative = nums.Length - 1;
        var firstPositive = 0;
        for (var i = 0; i < nums.Length; i++)
            if (nums[i] >= 0)
            {
                lastNegative = i - 1;
                break;
            }

        for (var i = nums.Length - 1; i >= 0; i--)
            if (nums[i] <= 0)
            {
                firstPositive = i + 1;
                break;
            }

        return Math.Max(lastNegative + 1, nums.Length - firstPositive);
    }

    public static int NumberOfSubstrings(string word)
    {
        var count = 0;
        var abc = new int[word.Length][];
        for (var i = 0; i < word.Length; i++)
        {
            abc[i] = new int[3];
            abc[i][word[i] - 'a']++;
            if (i <= 0)
                continue;
            for (var j = 0; j < 3; j++)
                abc[i][j] += abc[i - 1][j];
        }

        var start = -1;
        for (var i = 0; i < word.Length; i++)
        {
            if (abc[i].All(x => x > 0))
            {
                count += word.Length - i;
                start = i;
                break;
            }
        }

        if (start == -1)
            return 0;

        var left = 0;
        for (var i = start; i < word.Length; i++)
        {
            if (abc[i].All(x => x > 0))
            {
                var a = abc[i][0];
                var b = abc[i][1];
                var c = abc[i][2];
                while (i - left >= 2 && a - abc[left][0] > 0 && b - abc[left][1] > 0 && c - abc[left][2] > 0)
                {
                    left++;
                    count++;
                }
            }
        }

        return count;
    }

    public long CountOfSubstrings(string word, int k)
    {
        var isVowel = new int[128];
        var freq = new int[128];
        var vowels = "aeiou";

        foreach (var v in vowels)
            isVowel[v] = 1;

        long response = 0;
        var consolants = 0;
        var vowelCount = 0;
        var extraLeft = 0;
        var left = 0;

        for (var right = 0; right < word.Length; right++)
        {
            if (isVowel[word[right]] == 1)
            {
                if (++freq[word[right]] == 1)
                    vowelCount++;
            }
            else
                consolants++;

            while (consolants > k)
            {
                if (isVowel[word[left]] == 1)
                {
                    if (--freq[word[left]] == 0)
                        vowelCount--;
                }
                else
                {
                    consolants--;
                }

                left++;
                extraLeft = 0;
            }

            while (vowelCount == 5 && consolants == k && left < right && isVowel[word[left]] == 1 &&
                   freq[word[left]] > 1)
            {
                extraLeft++;
                freq[word[left]]--;
                left++;
            }

            if (consolants == k && vowelCount == 5)
            {
                response += 1 + extraLeft;
            }
        }

        return response;
    }

    static bool IsVowel(char s)
    {
        return new[] { 'a', 'e', 'i', 'o', 'u' }.Contains(s);
    }

    static bool ContainsAllVowels(string word, int start, int end)
    {
        var vowels = new[] { 'a', 'e', 'i', 'o', 'u' };
        var dict = new HashSet<char>();
        for (var i = start; i <= end; i++)
            if (vowels.Contains(word[i]))
                dict.Add(word[i]);
        return dict.Count == 5;
    }

    public static int NumberOfAlternatingGroups(int[] colors, int k)
    {
        var groups = 0;
        var n = colors.Length;

        var lengths = new HashSet<string>();
        var sb = new StringBuilder();
        var ends = new List<int>();
        for (var i = -k; i < n + k - 1; i++)
        {
            var first = (n + i) % n;
            var second = (n + i + 1) % n;
            if (colors[first] == colors[second])
            {
                if (sb.Length >= k)
                    lengths.Add(sb.ToString());
                sb = new StringBuilder();
                ends.Add(first);
                continue;
            }

            if (sb.Length == 0)
                sb.Append(first);
            sb.Append(second);
        }

        var s = 0;
        var x = new List<int>();
        if (ends.Count == 1)
        {
        }

        foreach (var e in ends)
        {
        }

        foreach (var l in lengths)
        {
            groups += l.Length - k + 1;
        }


        return groups;
    }

    public static int MinimumRecolors(string blocks, int k)
    {
        var whites = 0;
        for (var i = 0; i < k; i++)
            if (blocks[i] == 'W')
                whites++;
        var minWhites = whites;
        for (var i = k; i < blocks.Length; i++)
        {
            if (blocks[i - k] == 'W')
                whites--;
            if (blocks[i] == 'W')
                whites++;
            minWhites = Math.Min(whites, minWhites);
        }

        return minWhites;
    }

    public static int[] ClosestPrimes(int left, int right)
    {
        var composite = new bool[right + 1];
        Array.Fill(composite, true);
        var primes = 0;
        for (var i = 2; i < Math.Sqrt(right) + 1; i++)
        {
            if (!composite[i])
                continue;
            for (var j = i * i; j <= right; j += i)
            {
                composite[j] = false;
            }
        }

        var result = new List<int>();
        for (var i = Math.Max(2, left); i <= right; i++)
        {
            if (!composite[i])
                continue;
            result.Add(i);
        }

        var min = int.MaxValue;
        var a = -1;
        var b = -1;
        for (var i = 0; i < result.Count - 1; i++)
        {
            if (result[i + 1] - result[i] >= min)
                continue;
            if (result[i] == 0)
                continue;
            min = result[i + 1] - result[i];
            a = result[i];
            b = result[i + 1];
        }

        return new[] { a, b };
    }

    public int[] FindMissingAndRepeatedValues(int[][] grid)
    {
        var n = grid.Length;
        var buffer = new int[n * n];
        var sum = 0;
        var a = -1;
        var originalSum = (1 + n * n) * n * n / 2;
        for (var i = 0; i < n; i++)
        for (var j = 0; j < n; j++)
        {
            var x = grid[i][j];
            sum += x;
            if (buffer[x - 1] == 1)
                a = x;
            buffer[x - 1]++;
        }

        return new[] { a, originalSum - sum + a };
    }

    public long ColoredCells(int n)
    {
        long result = 0;
        for (var i = 1; i <= n; i++)
            result += Count(i);
        return result;
    }

    long Count(int n)
    {
        if (n == 1)
            return 1;
        if (n == 2)
            return 4;
        return 2 * n + 2 * (n - 2);
    }

    public static bool CheckPowersOfThree(int n)
    {
        while (n > 0)
        {
            if (n % 3 == 2)
                return false;
            n /= 3;
        }

        return true;
    }

    static string ToBaseThree(int n)
    {
        var builder = new StringBuilder(32);
        while (n > 0)
        {
            builder.Append(n % 3);
            n /= 3;
        }

        return builder.ToString();
    }

    public int[] PivotArray(int[] nums, int pivot)
    {
        var result = new int[nums.Length];
        var pointer = 0;
        foreach (var x in nums)
        {
            if (x >= pivot)
                continue;
            result[pointer] = x;
            pointer++;
        }

        foreach (var x in nums)
        {
            if (x != pivot)
                continue;
            result[pointer] = x;
            pointer++;
        }

        foreach (var x in nums)
        {
            if (x <= pivot)
                continue;
            result[pointer] = x;
            pointer++;
        }

        return result;
    }

    public int[][] MergeArrays(int[][] nums1, int[][] nums2)
    {
        var dict = new Dictionary<int, int>();
        foreach (var x in nums1.Concat(nums2))
        {
            if (!dict.ContainsKey(x[0]))
                dict.Add(x[0], x[1]);
            else
                dict[x[0]] += x[1];
        }

        return dict
            .OrderBy(x => x.Key)
            .Select(x => new[] { x.Key, x.Value })
            .ToArray();
    }

    public static int[] ApplyOperations(int[] nums)
    {
        for (var i = 0; i < nums.Length - 1; i++)
        {
            if (nums[i] != nums[i + 1])
                continue;
            nums[i] *= 2;
            nums[i + 1] = 0;
        }

        var zeroPosition = 0;
        for (var i = 0; i < nums.Length; i++)
        {
            if (nums[i] == 0)
                continue;
            zeroPosition = i;
            break;
        }

        return nums;
    }

    public static string ShortestCommonSupersequence(string str1, string str2)
    {
        var n = str1.Length;
        var m = str2.Length;
        var dp = new int[n + 1, m + 1];

        for (var i = 1; i <= n; i++)
        {
            for (var j = 1; j <= m; j++)
            {
                if (str1[i - 1] == str2[j - 1])
                    dp[i, j] = 1 + dp[i - 1, j - 1];
                else
                    dp[i, j] = Math.Max(dp[i - 1, j], dp[i, j - 1]);
            }
        }

        var builder = new StringBuilder("");
        var k = n;
        var l = m;

        while (k > 0 && l > 0)
        {
            if (str1[k - 1] == str2[l - 1])
            {
                builder.Append(str1[k - 1]);
                k--;
                l--;
            }
            else if (dp[k - 1, l] > dp[k, l - 1])
            {
                builder.Append(str1[k - 1]);
                k--;
            }
            else
            {
                builder.Append(str2[l - 1]);
                l--;
            }
        }

        while (k > 0)
        {
            builder.Append(str1[k - 1]);
            k--;
        }

        while (l > 0)
        {
            builder.Append(str2[l - 1]);
            l--;
        }

        return new string(builder.ToString().Reverse().ToArray());
    }

    static string LongestCommonSubsequence(string s1, string s2)
    {
        if (s1.Contains(s2))
            return s2;
        var sub = String.Empty;
        for (var i = 0; i < s2.Length; i++)
        {
            for (var j = 0; j < s2.Length; j++)
            {
                if (s2.Length - i - j < sub.Length)
                    continue;
                var subsequence = s2.Substring(i, s2.Length - i - j);
                if (s1.Contains(subsequence))
                    sub = subsequence;
            }
        }

        return sub;
    }

    public static int LenLongestFibSubseq(int[] arr)
    {
        var max = 0;
        var list = arr.ToList();
        for (var i = 0; i < arr.Length; i++)
        {
            var length = 2;
            var a = list[i];
            for (var j = i + 1; j < arr.Length; j++)
            {
                var b = list[j];
                var pointer = j + 1;
                while (true)
                {
                    var index = list.IndexOf(a + b, pointer);
                    if (index == -1)
                    {
                        a = list[i];
                        length = 2;
                        break;
                    }

                    a = b;
                    pointer = index + 1;
                    b = list[index];
                    length++;
                    max = Math.Max(max, length);
                }
            }
        }

        return max == 2 ? 0 : max;
    }

    public int MaxAbsoluteSum(int[] nums)
    {
        var currentMin = 0;
        var currentMax = 0;
        var max = 0;
        var min = 0;
        foreach (var x in nums)
        {
            currentMax = Math.Max(currentMax + x, x);
            currentMin = Math.Max(currentMin + x, x);
            max = Math.Max(max, currentMax);
            min = Math.Min(min, currentMin);
        }

        return Math.Max(max, Math.Abs(min));
    }

    public static int NumOfSubarrays(int[] arr)
    {
        long even = 1;
        long odd = 1;
        long result = 0;
        long sum = 0;
        foreach (var x in arr)
        {
            sum += x;
            if (sum % 2 == 0)
            {
                result = (result + odd) % 1000000007;
                even++;
            }
            else
            {
                result = (result + even) % 1000000007;
                odd++;
            }
        }

        return (int)result;
    }

    List<int>[] edges;
    Dictionary<int, int> bobPath = new();
    int[] amount;

    private int AlicePath(int curr, int currDay, int prev)
    {
        var result = 0;

        if (!bobPath.ContainsKey(curr))
            result += amount[curr];
        else if (bobPath[curr] > currDay)
            result += amount[curr];
        else if (bobPath[curr] == currDay)
            result += amount[curr] / 2;

        var max = int.MinValue;
        foreach (var edge in edges[curr])
        {
            if (edge == prev)
                continue;
            max = Math.Max(max, AlicePath(edge, currDay + 1, curr));
        }

        if (max == int.MinValue)
            return result;
        return result + max;
    }

    private bool BobPath(int current, int currDay, int prev)
    {
        bobPath[current] = currDay;
        if (current == 0)
            return true;

        foreach (var edge in edges[current])
        {
            if (edge == prev)
                continue;
            if (BobPath(edge, currDay + 1, current))
                return true;
        }

        bobPath.Remove(current);
        return false;
    }

    static void BobPath(int[] bobTime)
    {
    }

    public TreeNode ConstructFromPrePost(int[] preorder, int[] postorder)
    {
        postIndex = new();
        for (var i = 0; i < preorder.Length; i++)
            postIndex[preorder[i]] = i;
        var n = preorder.Length;
        return BuildTree(0, n - 1, n - 1, 0, preorder);
    }

    private static Dictionary<int, int> postIndex;

    TreeNode BuildTree(int preStart, int preEnd, int postStart, int postEnd,
        int[] preorder)
    {
        if (preStart > preEnd || postStart > postEnd)
            return null;
        var root = new TreeNode(preorder[preStart]);
        if (preStart != preEnd)
        {
            var mid = postIndex[preorder[preStart + 1]];
            var size = mid - postStart + 1;
            root.left = BuildTree(preStart + 1, preStart + size, postStart, mid, preorder);
            root.right = BuildTree(preStart + size + 1, preEnd, mid + 1, postEnd - 1, preorder);
        }

        return root;
    }

    static void Preorder(TreeNode node)
    {
        if (node == null)
            return;
        Preorder(node.left);
        Preorder(node.right);
    }

    static void Postorder(TreeNode node)
    {
        if (node == null)
            return;
        Postorder(node.left);
        Postorder(node.right);
    }

    public static TreeNode RecoverFromPreorder(string traversal)
    {
        var levels = new List<TreeNode>();
        for (var i = 0; i < traversal.Length; i++)
        {
            var level = 0;
            while (i < traversal.Length && traversal[i] == '-')
            {
                i++;
                level++;
            }

            var sb = new StringBuilder();
            while (i < traversal.Length && char.IsDigit(traversal[i]))
            {
                sb.Append(traversal[i]);
                i++;
            }

            var value = int.Parse(sb.ToString());
            var node = new TreeNode(value);
            if (level < levels.Count)
                levels[level] = node;
            else
                levels.Add(node);
            if (level > 0)
            {
                var parent = levels[level - 1];
                if (parent.left != null)
                    parent.right = node; // this is the .right child
                else
                    parent.left = node; // otherwise it is the .left child
            }

            i--;
        }

        return levels[0];
    }

    public static string FindDifferentBinaryString(string[] nums)
    {
        var n = nums[0].Length;
        var array = new int[nums.Length];
        for (var i = 0; i < nums.Length; i++)
        {
            for (var j = 0; j < n; j++)
            {
                if (nums[i][j] == '0')
                    continue;
                array[i] += 1 << (n - 1 - j);
            }
        }

        Array.Sort(array);
        for (var i = 0; i < array.Length - 1; i++)
        {
            if (array[0] > 0)
                return ToBinary(0, n);
            if (array[i + 1] - array[i] > 1)
            {
                return ToBinary(array[i] + 1, n);
            }
        }

        return ToBinary(array[^1] + 1, n);
    }

    static string ToBinary(int value, int length)
    {
        var array = new char[length];
        for (var i = 0; i < length; i++)
            array[length - 1 - i] = (value & (1 << i)) >> i == 1 ? '1' : '0';

        return new string(array);
    }

    private static HashSet<string> meow = new();

    public static string GetHappyString(int n, int k)
    {
        var board = new char[n];
        Array.Fill(board, '\0');
        count = 0;
        var v = BackTrackHappyStrings(0, board, k);
        return v;
    }

    private static int count;

    static string BackTrackHappyStrings(int position, char[] board, int k)
    {
        if (position == board.Length)
        {
            count++;
            return new string(board);
        }

        if (position == 0)
            count++;
        var solution = new List<char> { 'a', 'b', 'c' };
        if (position > 0)
            solution.Remove(board[position - 1]);
        foreach (var symbol in solution)
        {
            board[position] = symbol;
            var v = BackTrackHappyStrings(position + 1, board, k);
            if (count == k + 1)
                return v;
            board[position] = '\0';
        }

        return string.Empty;
    }

    public static string SmallestNumber(string pattern)
    {
        var stack = new Stack<char>();
        var builder = new StringBuilder(pattern.Length + 1);

        for (var i = 1; i <= pattern.Length + 1; i++)
        {
            stack.Push((char)(i + '0'));
            if (i != pattern.Length + 1 && pattern[i - 1] != 'I')
                continue;
            while (stack.Count > 0)
                builder.Append(stack.Pop());
        }

        return builder.ToString();
    }

    static List<string> permutations = new();

    static void MakePermutations(int[] permutation, int position, string pattern)
    {
        if (position == permutation.Length)
        {
            var x = string.Join("", permutation);
            var found = true;
            for (var i = 0; i < x.Length - 1; i++)
            {
                if (pattern[i] == 'D')
                {
                    if (x[i] <= x[i + 1])
                    {
                        found = false;
                        break;
                    }
                }
                else if (pattern[i] == 'I')
                {
                    if (x[i] >= x[i + 1])
                    {
                        found = false;
                        break;
                    }
                }
            }

            if (found)
                permutations.Add(x);
            return;
        }

        for (var i = 1; i <= permutation.Length; i++)
        {
            var index = Array.IndexOf(permutation, i, 0, position);
            if (index != -1)
                continue;
            permutation[position] = i;
            MakePermutations(permutation, position + 1, pattern);
        }
    }

    static void BackTracking2(int position, int[] board, HashSet<int> alphabet, string pattern)
    {
        if (position == board.Length)
        {
            meow.Add(string.Join("", board));
            return;
        }

        if (board[position] != 0)
            BackTracking2(position + 1, board, alphabet, pattern);
        var solution = Enumerable.Range(1, board.Length)
            .Where(x => !board.Contains(x))
            .ToArray();
        foreach (var symbol in solution)
        {
            board[position] = symbol;
            BackTracking2(position + 1, board, alphabet, pattern);
            board[position] = 0;
            ;
        }
    }

    static int TwoNumbers(int target, int source)
    {
        var count = 0;
        while (source > target)
        {
            if (source % 2 == 0)
                source /= 2;
            else
                source++;
            count++;
        }

        return Math.Abs(source - target) + count;
    }

    public string MostCommonWord(string paragraph, string[] banned)
    {
        var words = new List<string>();
        var sb = new StringBuilder();
        foreach (var x in paragraph)
        {
            if (char.IsLetter(x))
            {
                sb.Append(x);
            }
            else if (sb.Length > 0)
            {
                words.Add(sb.ToString());
                sb = new StringBuilder();
            }
        }

        if (sb.Length != 0)
            words.Add(sb.ToString());
        var dict = words
            .Select(x => x.ToLower())
            .Where(x => !banned.Contains(x))
            .GroupBy(x => x)
            .Select(x => new { Word = x.Key, Count = x.Count() })
            .OrderByDescending(x => x.Count)
            .ToArray();
        return dict.FirstOrDefault()?.Word;
    }

    public static int NumTilePossibilities(string tiles)
    {
        var alphabet = tiles.GroupBy(x => x)
            .ToDictionary(x => x.Key, x => x.Count());
        for (var l = 1; l <= tiles.Length; l++)
        {
            BackTracking1(0, new char[l], l, alphabet);
        }

        return meow.Count;
    }

    static void BackTracking1(int position, char[] board, int length, Dictionary<char, int> alphabet)
    {
        if (position == length)
        {
            meow.Add(new string(board));
            return;
        }

        if (board[position] != '\0')
            BackTracking1(position + 1, board, length, alphabet);
        foreach (var symbol in alphabet.Keys)
        {
            if (alphabet[symbol] == 0)
                continue;
            board[position] = symbol;
            alphabet[symbol]--;
            BackTracking1(position + 1, board, length, alphabet);
            board[position] = '\0';
            alphabet[symbol]++;
        }
    }

    public int[] ConstructDistancedSequence(int n)
    {
        var length = 2 * (n - 1) + 1;
        var result = new int[length];
        Array.Fill(result, -1);
        BackTracking(0, result, n);
        for (var i = 0; i < length; i++)
            if (result[i] == 0 || result[i] == -1)
            {
                result[i] = 1;
                break;
            }

        return result;
    }

    bool BackTracking(int position, int[] board, int n)
    {
        if (position == board.Length - 1)
        {
            return true;
        }

        if (board[position] != -1)
            return BackTracking(position + 1, board, n);
        var solution = Enumerable.Range(2, n - 1).Except(board.Where(x => x != 0)).OrderByDescending(x => x).ToList();
        if (!board.Contains(0))
            solution.Add(0);
        foreach (var i in solution)
        {
            if (position + i > board.Length - 1 || board[position + i] != -1)
                continue;
            board[position] = i;
            board[position + i] = i;
            if (BackTracking(position + 1, board, n))
                return true;
            board[position] = -1;
            board[position + i] = -1;
        }

        return false;
    }

    public static int PunishmentNumber(int n)
    {
        var result = 1;
        for (var i = 2; i <= n; i++)
        {
            var square = i * i;
            var values = square.ToString().Select(x => x - '0').ToArray();


            if (HasSum(values, i))
                result += i * i;
        }

        return result;
    }

    static bool HasSum(int[] values, int k, int currentSum = 0)
    {
        if (!values.Any())
            return currentSum == k;
        for (var p = 1; p <= values.Length; p++)
        {
            var t = values.Take(p).Select(x => (char)(x + '0')).ToArray();
            var leftSum = int.Parse(new string(t));
            var subSequence = values.Skip(p).ToArray();
            var found = HasSum(subSequence, k, leftSum + currentSum);
            if (found)
                return true;
        }

        return false;
    }

    public static int MinOperations1(int[] nums, int k)
    {
        var queue = new PriorityQueue<long, long>();
        foreach (var x in nums)
        {
            queue.Enqueue(x, x);
        }

        var count = 0;

        while (queue.Count >= 2)
        {
            var x = queue.Dequeue();
            if (x >= k)
                return count;
            var y = queue.Dequeue();
            var value = 2 * x + y;
            queue.Enqueue(value, value);
            count++;
        }

        return count;
    }

    public static int MaximumSum(int[] nums)
    {
        var dict = new Dictionary<int, (int Min, int Max)>();
        var maxSum = -1;
        foreach (var x in nums)
        {
            var sumDigits = GetSumDigits(x);
            if (!dict.ContainsKey(sumDigits))
                dict.Add(sumDigits, (0, 0));
            var min = Math.Min(dict[sumDigits].Min, dict[sumDigits].Max);
            var max = Math.Max(dict[sumDigits].Min, dict[sumDigits].Max);
            dict[sumDigits] = (min, max);
            dict[sumDigits] = (Math.Max(min, x), dict[sumDigits].Max);
            if (dict[sumDigits].Min != 0 && dict[sumDigits].Max != 0)
                maxSum = Math.Max(maxSum, dict[sumDigits].Min + dict[sumDigits].Max);
        }

        return maxSum;
    }

    static int GetSumDigits(int x)
    {
        var sum = 0;
        while (x > 0)
        {
            sum += x % 10;
            x /= 10;
        }

        return sum;
    }

    public static string RemoveOccurrences(string s, string part)
    {
        var word = s;
        while (true)
        {
            var sb = new StringBuilder(word);
            var index = word.IndexOf(part);
            if (index == -1)
                return word;
            for (var i = 0; i < part.Length; i++)
                sb[index + i] = '\0';
            var result = new StringBuilder();
            for (var i = 0; i < sb.Length; i++)
                if (sb[i] != '\0')
                    result.Append(sb[i]);
            word = result.ToString();
        }
    }

    public static string ClearDigits(string s)
    {
        var sb = new StringBuilder(s);
        var letterIndex = -1;
        for (var i = s.Length - 1; i >= 0; i--)
        {
            if (char.IsDigit(s[i]))
                break;
        }

        var result = new StringBuilder();
        for (var i = 0; i < s.Length; i++)
        {
            if (sb[i] == '\0')
                continue;
            result.Append(sb[i]);
        }

        return result.ToString();
    }

    public static bool CheckStraightLine(int[][] coordinates)
    {
        var (x1, y1) = (coordinates[0][0], coordinates[0][1]);
        var (x2, y2) = (coordinates[1][0], coordinates[1][1]);
        for (var i = 1; i < coordinates.Length; i++)
        {
            var (x, y) = (coordinates[i][0], coordinates[i][1]);
            if ((x - x1) * (y2 - y1) != (y - y1) * (x2 - x1))
                return false;
        }

        return true;
    }

    public static long CountBadPairs(int[] nums)
    {
        var frequency = new Dictionary<int, int>();
        for (var i = 0; i < nums.Length; i++)
        {
            if (!frequency.ContainsKey(nums[i] - i))
                frequency.Add(nums[i] - i, 0);
            frequency[nums[i] - i]++;
        }

        if (frequency.Keys.Count == 1)
            return 0;
        var n = nums.Length;
        var result = (long)n * (n - 1) / 2;
        foreach (var f in frequency.Values.Where(x => x > 1))
            result -= (long)f * (f - 1) / 2;
        return result;
    }

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

        static char[] Vowels = { 'a', 'e', 'i', 'o', 'u' };

        public static int[] QueryResults(int limit, int[][] queries)
        {
            var ballToColor = new Dictionary<int, int>();
            var colorToBalls = new Dictionary<int, HashSet<int>>();
            var result = new int[queries.Length];
            for (var i = 0; i < queries.Length; i++)
            {
                var q = queries[i];
                var ball = q[0];
                var color = q[1];

                if (!ballToColor.ContainsKey(ball))
                    ballToColor.Add(ball, 0);
                else
                {
                    var deletedColor = ballToColor[ball];
                    colorToBalls[deletedColor].Remove(ball);
                    if (colorToBalls[deletedColor].Count == 0)
                        colorToBalls.Remove(deletedColor);
                }

                ballToColor[ball] = color;
                if (!colorToBalls.ContainsKey(color))
                    colorToBalls.Add(color, new HashSet<int>());
                colorToBalls[color].Add(ball);
                result[i] = colorToBalls.Count;
            }

            return result;
        }

        public static int TupleSameProduct(int[] nums)
        {
            var n = nums.Length;
            var frequency = new Dictionary<int, int>();
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < n; j++)
                {
                    if (i == j)
                        continue;
                    var value = nums[i] * nums[j];
                    frequency.TryAdd(value, 0);
                    frequency[value]++;
                }
            }

            //todo тут считаются перестановки (A от n/2 элементов по 2)* 2
            // n/2 - количество различных элементов, образующих равные произведения
            // * 2 это все - потому что мы и произведения двух произведений
            // можем переставить местами, образовав верное тождество

            var count = 0;
            foreach (var x in frequency.Values)
            {
                if (x < 4)
                    continue;
                var value = x * (x - 1);
                count += value;
            }

            return count;
        }

        public static bool AreAlmostEqual(string s1, string s2)
        {
            if (s1 == s2)
                return true;
            var notEqual = new int[2];
            var pointer = 0;
            for (var i = 0; i < s1.Length; i++)
            {
                if (s1[i] != s2[i])
                {
                    if (pointer == 2)
                        return false;
                    notEqual[pointer] = i;
                    pointer++;
                }
            }

            if (s2[notEqual[1]] == s1[notEqual[0]] && s2[notEqual[0]] == s1[notEqual[1]])
                return true;
            return false;
        }

        public int MaxAscendingSum(int[] nums)
        {
            var currentSum = nums[0];
            var max = currentSum;
            for (var i = 0; i < nums.Length - 1; i++)
            {
                if (nums[i] < nums[i + 1])
                {
                    currentSum += nums[i + 1];
                    max = Math.Max(max, currentSum);
                }
                else
                {
                    currentSum = nums[i + 1];
                }
            }

            return max;
        }

        public static int LongestMonotonicSubarray(int[] nums)
        {
            var max = 1;
            var current = 1;
            for (var i = 0; i < nums.Length - 1; i++)
            {
                if (nums[i + 1] <= nums[i])
                {
                    current = 1;
                }
                else
                {
                    current++;
                }

                max = Math.Max(max, current);
            }

            current = 1;
            for (var i = 0; i < nums.Length - 1; i++)
            {
                if (nums[i + 1] >= nums[i])
                {
                    current = 1;
                }
                else
                {
                    current++;
                }

                max = Math.Max(max, current);
            }

            return max;
        }

        public static bool Check1(int[] nums)
        {
            var n = nums.Length;
            var min = int.MaxValue;
            var index = -1;
            for (var i = 0; i < n; i++)
            {
                if (nums[i] >= min)
                    continue;
                min = nums[i];
                index = i;
            }

            for (var i = 0; i < n; i++)
            {
                if (nums[(n + index - 1) % n] != min)
                    break;
                index--;
                index = (n + index) % n;
            }

            for (var i = 0; i < n - 1; i++)
            {
                if (nums[(index + i) % n] > nums[(index + i + 1) % n])
                    return false;
            }

            return true;
        }

        public bool IsArraySpecial(int[] nums)
        {
            for (var i = 0; i < nums.Length - 1; i++)
                if ((nums[i] + nums[i + 1]) % 2 == 0)
                    return false;
            return true;
        }

        public static int LargestIsland(int[][] grid)
        {
            var height = grid.Length;
            var width = grid[0].Length;
            var maxSize = 0;
            var islandSize = new Dictionary<(int X, int Y), int>();
            var islandColor = new Dictionary<(int X, int Y), int>();
            var allVisited = new HashSet<(int X, int Y)>();
            var color = 0;
            var foundIsland = false;
            for (var y = 0; y < height; y++)
            {
                for (var x = 0; x < width; x++)
                {
                    if (grid[y][x] == 0)
                        continue;
                    foundIsland = true;
                    if (allVisited.Contains((x, y)))
                    {
                        continue;
                    }

                    color++;
                    var size = 0;
                    var visited = new HashSet<(int X, int Y)>();
                    var queue = new Queue<(int X, int Y)>();
                    queue.Enqueue((x, y));
                    while (queue.Count > 0)
                    {
                        var (xi, yi) = queue.Dequeue();
                        if (!visited.Add((xi, yi)))
                            continue;
                        size++;
                        if (xi - 1 >= 0 && !visited.Contains((xi - 1, yi)) && grid[yi][xi - 1] == 1)
                            queue.Enqueue((xi - 1, yi));
                        if (xi + 1 < width && !visited.Contains((xi + 1, yi)) && grid[yi][xi + 1] == 1)
                            queue.Enqueue((xi + 1, yi));
                        if (yi - 1 >= 0 && !visited.Contains((xi, yi - 1)) && grid[yi - 1][xi] == 1)
                            queue.Enqueue((xi, yi - 1));
                        if (yi + 1 < height && !visited.Contains((xi, yi + 1)) && grid[yi + 1][xi] == 1)
                            queue.Enqueue((xi, yi + 1));
                    }

                    foreach (var node in visited)
                    {
                        allVisited.Add(node);
                        islandSize.Add(node, size);
                        islandColor.Add(node, color);
                    }
                }
            }

            if (!foundIsland)
                return 1;

            //todo trying to merge 2 islands
            for (var y = 0; y < height; y++)
            {
                for (var x = 0; x < width; x++)
                {
                    if (grid[y][x] == 1)
                        continue;
                    var islands = new List<(int X, int Y)>();
                    if (x - 1 >= 0 && grid[y][x - 1] == 1)
                    {
                        islands.Add((x - 1, y));
                    }

                    if (x + 1 < width && grid[y][x + 1] == 1)
                    {
                        islands.Add((x + 1, y));
                    }

                    if (y - 1 >= 0 && grid[y - 1][x] == 1)
                    {
                        islands.Add((x, y - 1));
                    }

                    if (y + 1 < height && grid[y + 1][x] == 1)
                    {
                        islands.Add((x, y + 1));
                    }

                    var s = islands.Select(i => new { Size = islandSize[i], Color = islandColor[i] })
                        .DistinctBy(i => i.Color)
                        .Select(i => i.Size)
                        .Sum();
                    maxSize = Math.Max(maxSize, s + 1);
                }
            }

            return maxSize == 0 ? height * width : maxSize;
        }

        // public int MagnificentSets(int n, int[][] edges)
        // {
        //     var total = 0;
        //     var dict = new List<List<int>>();
        //     for (var i = 0; i <= n; i++)
        //         dict.Add(new List<int>());
        //     foreach (var x in edges)
        //     {
        //         dict[x[0]].Add(x[1]);
        //         dict[x[1]].Add(x[0]);
        //     }
        //
        //     var visited = new HashSet<int>();
        //     var colors = new int[n + 1];
        //     for (var x = 1; i <= n; i++)
        //     {
        //         if (visited.Contains(x))
        //             continue;
        //         var component = new List<int>();
        //         if (!IsBipartite(x, dict, colors, visited, component))
        //             return -1;
        //
        //         var maxGroups = 0;
        //         foreach (var y in component)
        //         {
        //             maxGroups = Math.Max(maxGroups, Bfs(y, dict));
        //         }
        //
        //         total += maxGroups;
        //     }
        //
        //     return total;
        // }

        bool IsBipartite(int start, List<List<int>> dict, int[] colors, HashSet<int> visited,
            List<int> component)
        {
            Queue<int> queue = new Queue<int>();
            queue.Enqueue(start);
            colors[start] = 1;
            visited.Add(start);

            while (queue.Count > 0)
            {
                int node = queue.Dequeue();
                component.Add(node);

                foreach (int neighbor in dict[node])
                {
                    if (colors[neighbor] == 0)
                    {
                        colors[neighbor] = -colors[node];
                        visited.Add(neighbor);
                        queue.Enqueue(neighbor);
                    }
                    else if (colors[neighbor] == colors[node])
                    {
                        return false;
                    }
                }
            }

            return true;
        }

        int Bfs(int x, List<List<int>> dict)
        {
            var queue = new Queue<int>();
            queue.Enqueue(x);
            var depths = new Dictionary<int, int>();
            depths[x] = 1;
            var maxDepth = 1;
            while (queue.Count > 0)
            {
                var node = queue.Dequeue();
                var depth = depths[node];
                foreach (int neighbor in dict[node])
                {
                    if (!depths.ContainsKey(neighbor))
                    {
                        depths[neighbor] = depth + 1;
                        maxDepth = Math.Max(maxDepth, depth + 1);
                        queue.Enqueue(neighbor);
                    }
                }
            }

            return maxDepth;
        }

        public int[] FindRedundantConnection(int[][] edges)
        {
            for (var j = edges.Length - 1; j >= 0; j--)
            {
                var graph = Enumerable.Range(0, edges.Length)
                    .Where(x => x != j)
                    .Select(x => edges[x])
                    .ToArray();
                var neighbours = graph.SelectMany(x => new[]
                    {
                        (x[0], x[1]),
                        (x[1], x[0]),
                    })
                    .GroupBy(x => x.Item1)
                    .ToDictionary(x => x.Key, x => x.Select(y => y.Item2).ToArray());
                var visited = new HashSet<int>();
                var queue = new Queue<int>();
                queue.Enqueue(graph.First()[0]);
                var count = 0;
                while (queue.Count > 0)
                {
                    var node = queue.Dequeue();
                    if (!visited.Add(node))
                        continue;
                    count++;
                    foreach (var n in neighbours[node])
                    {
                        if (!visited.Contains(n) && !queue.Contains(n))
                            queue.Enqueue(n);
                    }
                }

                if (count == edges.Length)
                    return edges[j];
            }

            return Array.Empty<int>();
        }

        public static string ReorderSpaces(string text)
        {
            var words = new List<string>();
            var builder = new StringBuilder(text.Length);
            var whitespaces = 0;
            foreach (var x in text)
            {
                if (x == ' ')
                {
                    whitespaces++;
                    if (builder.Length == 0)
                        continue;
                    words.Add(builder.ToString());
                    builder.Clear();
                }
                else
                    builder.Append(x);
            }

            if (builder.Length != 0)
                words.Add(builder.ToString());

            var step = words.Count > 1 ? whitespaces / (words.Count - 1) : 0;
            return string.Join(new string(' ', step), words) +
                   new string(' ', words.Count > 1 ? whitespaces % (words.Count - 1) : whitespaces);
        }

        public static int FindMaxFish(int[][] grid)
        {
            var height = grid.Length;
            var width = grid[0].Length;
            var visited = new HashSet<(int X, int Y)>();
            var maxScore = 0;
            for (var y = 0; y < height; y++)
            {
                for (var x = 0; x < width; x++)
                {
                    if (grid[y][x] == 0)
                        continue;
                    var queue = new Queue<(int X, int Y)>();
                    queue.Enqueue((x, y));
                    var score = 0;
                    while (queue.Count > 0)
                    {
                        var (xi, yi) = queue.Dequeue();
                        if (!visited.Add((xi, yi)))
                            continue;
                        score += grid[yi][xi];

                        if (xi - 1 >= 0 && grid[yi][xi - 1] != 0 && !visited.Contains((xi - 1, yi)))
                            queue.Enqueue((xi - 1, yi));
                        if (xi + 1 < width && grid[yi][xi + 1] != 0 && !visited.Contains((xi + 1, yi)))
                            queue.Enqueue((xi + 1, yi));
                        if (yi - 1 >= 0 && grid[yi - 1][xi] != 0 && !visited.Contains((xi, yi - 1)))
                            queue.Enqueue((xi, yi - 1));
                        if (yi + 1 < height && grid[yi + 1][xi] != 0 && !visited.Contains((xi, yi + 1)))
                            queue.Enqueue((xi, yi + 1));
                    }

                    maxScore = Math.Max(maxScore, score);
                }
            }

            return maxScore;
        }

        public static IList<bool> CheckIfPrerequisite(int numCourses, int[][] prerequisites, int[][] queries)
        {
            var reached = new bool[numCourses][];
            for (var i = 0; i < numCourses; i++)
                reached[i] = new bool[numCourses];
            foreach (var prerequisite in prerequisites)
            {
                var first = prerequisite[0];
                var next = prerequisite[1];
                reached[first][next] = true;
            }

            for (var i = 0; i < numCourses; i++)
            {
                for (var j = 0; j < numCourses; j++)
                {
                    for (var k = 0; k < numCourses; k++)
                        reached[j][k] = reached[j][k] || (reached[j][i] && reached[i][k]);
                }
            }

            return queries.Select(x => reached[x[0]][x[1]]).ToArray();
        }

        public bool CanAliceWin(int n)
        {
            var step = 10;
            for (var i = 0; i < 10; i++)
            {
                if (n < step)
                    return i % 2 != 0;
                n -= step--;
            }

            return false;
        }

        public int MaximumInvitations(int[] favorite)
        {
            var n = favorite.Length;
            var likes = new int[n];

            for (var i = 0; i < n; i++)
                likes[favorite[i]]++;
            var unrequited = new Queue<int>(n);
            for (var i = 0; i < n; i++)
                if (likes[i] == 0)
                    unrequited.Enqueue(i);
            var chains = new int[n];
            var visited = new HashSet<int>(n);
            while (unrequited.Count > 0)
            {
                var unwanted = unrequited.Dequeue();
                var person = favorite[unwanted];
                chains[person] = Math.Max(chains[person], chains[unwanted] + 1);
                likes[person]--;
                if (likes[person] == 0)
                    unrequited.Enqueue(person);
            }

            var maxChain = 0;
            var maxCycle = 0;
            for (var i = 0; i < n; i++)
            {
                if (!visited.Contains(i) && likes[i] > 0)
                {
                    var cycle = 1;
                    visited.Add(i);
                    var person = favorite[i];
                    while (person != i)
                    {
                        visited.Add(person);
                        person = favorite[person];
                        cycle++;
                    }

                    if (cycle == 2)
                    {
                        var x = i;
                        var y = favorite[i];
                        maxChain += chains[y] + chains[x] + 2;
                    }
                    else
                    {
                        maxCycle = Math.Max(maxCycle, cycle);
                    }
                }
            }

            return Math.Max(maxChain, maxCycle);
        }

        public static int[] LexicographicallySmallestArray(int[] nums, int limit)
        {
            var sorted = nums.OrderBy(x => x).ToArray();
            var groupIndexes = new Dictionary<int, int>();
            var groups = new Dictionary<int, Queue<int>>();
            var currentGroup = 0;
            groupIndexes.Add(sorted[currentGroup], currentGroup);
            groups.Add(currentGroup, new Queue<int>());
            groups[currentGroup].Enqueue(sorted[currentGroup]);
            for (var i = 1; i < sorted.Length; i++)
            {
                if (sorted[i] - sorted[i - 1] > limit)
                {
                    currentGroup++;
                    groups.Add(currentGroup, new Queue<int>());
                }

                groupIndexes[sorted[i]] = currentGroup;
                groups[currentGroup].Enqueue(sorted[i]);
            }

            for (var i = 0; i < nums.Length; i++)
            {
                var group = groupIndexes[nums[i]];
                nums[i] = groups[group].Dequeue();
            }

            return nums;
        }

        public int[] FindErrorNums(int[] nums)
        {
            var frequency = new int[nums.Length];
            foreach (var x in nums)
                frequency[x - 1]++;
            var result = new int[2];
            var pointer = 0;
            for (var i = 0; i < frequency.Length; i++)
            {
                if (frequency[i] == 2 || frequency[i] == 0)
                    result[pointer++] = i + 1;
            }

            if (result[0] > result[1])
                (result[0], result[1]) = (result[1], result[0]);
            return result;
        }

        public static IList<int> EventualSafeNodes(int[][] graph)
        {
            var terminal = new int[graph.Length];
            for (var i = 0; i < graph.Length; i++)
            {
                if (graph[i].Length == 0)
                    terminal[i] = 2;
            }

            var result = new List<int>(graph.Length);
            for (var i = 0; i < graph.Length; i++)
            {
                if (IsSafe(graph, i, terminal))
                    result.Add(i);
            }

            return result;
        }

        static bool IsSafe(int[][] graph, int v, int[] terminal)
        {
            if (terminal[v] > 0)
                return terminal[v] == 2;
            terminal[v] = 1;
            foreach (var n in graph[v])
            {
                if (!IsSafe(graph, n, terminal))
                    return false;
            }

            terminal[v] = 2;
            return true;
        }

        public static int CountServers(int[][] grid)
        {
            var height = grid.Length;
            var width = grid[0].Length;
            var connected = 0;
            var rows = new int[height];
            var columns = new int[width];
            for (var y = 0; y < height; y++)
            {
                for (var x = 0; x < width; x++)
                {
                    if (grid[y][x] == 0)
                        continue;
                    columns[x]++;
                    rows[y]++;
                }
            }

            for (var y = 0; y < height; y++)
            {
                for (var x = 0; x < width; x++)
                {
                    if (grid[y][x] == 0)
                        continue;
                    if (rows[y] > 1 || columns[x] > 1)
                        connected++;
                }
            }

            return connected;
        }

        public static int[][] UpdateMatrix(int[][] map)
        {
            var height = map.Length;
            var width = map[0].Length;
            var answer = new int[height][];
            for (var y = 0; y < height; y++)
            {
                answer[y] = new int[width];
                Array.Fill(answer[y], -1);
            }

            var queue = new Queue<(int X, int Y)>();
            for (var y = 0; y < height; y++)
            for (var x = 0; x < width; x++)
            {
                if (map[y][x] == 1)
                    continue;
                answer[y][x] = 0;
                queue.Enqueue((x, y));
            }

            while (queue.Count > 0)
            {
                var (x, y) = queue.Dequeue();
                if (x - 1 >= 0 && answer[y][x - 1] == -1)
                {
                    queue.Enqueue((x - 1, y));
                    answer[y][x - 1] = answer[y][x] + 1;
                }

                if (x + 1 < width && answer[y][x + 1] == -1)
                {
                    answer[y][x + 1] = answer[y][x] + 1;
                    queue.Enqueue((x + 1, y));
                }

                if (y - 1 >= 0 && answer[y - 1][x] == -1)
                {
                    answer[y - 1][x] = answer[y][x] + 1;
                    queue.Enqueue((x, y - 1));
                }

                if (y + 1 < height && answer[y + 1][x] == -1)
                {
                    answer[y + 1][x] = answer[y][x] + 1;
                    queue.Enqueue((x, y + 1));
                }
            }

            return answer;
        }

        public long GridGame(int[][] grid)
        {
            var row0 = grid[0];
            var row1 = grid[1];
            var l = 0;
            var r = row0.Length - 1;
            var nextLeftSum = row1[l];
            var nextRightSum = row0[r];
            var leftSum = 0;
            var rightSum = 0;
            while (l < r)
            {
                if (nextLeftSum < nextRightSum)
                {
                    leftSum = nextLeftSum;
                    nextLeftSum += row1[++l];
                }
                else
                {
                    rightSum = nextRightSum;
                    nextRightSum += row0[--r];
                }
            }

            return Math.Max(leftSum, rightSum);
        }

        public static int MaxRepeating(string sequence, string word)
        {
            var left = 0;
            var maxCount = 0;
            for (var right = 0; right < sequence.Length; right++)
            {
                var count = 0;
                while (left + word.Length <= sequence.Length && sequence[left] != word[0])
                {
                    left++;
                }

                while (left + word.Length <= sequence.Length && sequence.Substring(left, word.Length) == word)
                {
                    count++;
                    left += word.Length;
                }

                maxCount = Math.Max(maxCount, count);
            }

            return maxCount;
        }

        public string TriangleType(int[] nums)
        {
            var (a, b, c) = (nums[0], nums[1], nums[2]);
            if (a + b <= c || a + c <= b || b + c <= a)
                return "none";
            if (a == b && b == c)
                return "equilateral";
            if (a == b || a == c || b == c)
                return "isosceles";
            return "scalene";
        }

        public bool IsBoomerang(int[][] points)
        {
            var (x1, y1) = (points[0][0], points[0][1]);
            var (x2, y2) = (points[1][0], points[1][1]);
            var (x3, y3) = (points[2][0], points[2][1]);
            if ((x1 == x2 && y1 == y2) || (x1 == x3 && y1 == y3) || (x1 == x3 && y1 == y3))
                return false;
            return x2 * y3 - x1 * y3 != x3 * y2 - x3 * y1 + x2 * y1 - x1 * y2;
        }

        public int FirstCompleteIndex(int[] arr, int[][] map)
        {
            var height = map.Length;
            var width = map[0].Length;
            var rows = new int[height];
            var columns = new int[width];
            var dict = new (int, int Y)[width * height + 1];
            for (var y = 0; y < height; y++)
            for (var x = 0; x < width; x++)
                dict[map[y][x]] = (x, y);
            for (var i = 0; i < arr.Length; i++)
            {
                var (x, y) = dict[arr[i]];
                rows[y]++;
                columns[x]++;
                if (rows[y] == width || columns[x] == height)
                    return i;
            }

            return -1;
        }

        public static int TrapRainWater(int[][] heightMap)
        {
            var height = heightMap.Length;
            var width = heightMap[0].Length;
            var priorityQueue = new PriorityQueue<(int x, int y, int val), int>
                (Comparer<int>.Create((a, b) => a.CompareTo(b)));
            for (var y = 0; y < height; y++)
            {
                priorityQueue.Enqueue((y, 0, heightMap[y][0]), heightMap[y][0]);
                heightMap[y][0] = -1;
                priorityQueue.Enqueue((y, width - 1, heightMap[y][width - 1]), heightMap[y][width - 1]);
                heightMap[y][width - 1] = -1;
            }

            for (var y = 1; y < width - 1; y++)
            {
                priorityQueue.Enqueue((0, y, heightMap[0][y]), heightMap[0][y]);
                heightMap[0][y] = -1;
                priorityQueue.Enqueue((height - 1, y, heightMap[height - 1][y]), heightMap[height - 1][y]);
                heightMap[height - 1][y] = -1;
            }

            var dir = new (int dx, int dy)[] { (0, 1), (0, -1), (1, 0), (-1, 0) };
            var volume = 0;
            while (priorityQueue.Count > 0)
            {
                var (x, y, h) = priorityQueue.Dequeue();
                foreach (var (dx, dy) in dir)
                {
                    var nx = x + dx;
                    var ny = y + dy;
                    if (nx < 0 || nx >= height || ny < 0 || ny >= width || heightMap[nx][ny] == -1)
                        continue;
                    volume += Math.Max(0, h - heightMap[nx][ny]);
                    var newHeight = Math.Max(heightMap[nx][ny], h);
                    priorityQueue.Enqueue((nx, ny, newHeight), newHeight);
                    heightMap[nx][ny] = -1;
                }
            }

            return volume;
        }

        static (int x, int y)[] BreadthSearch(
            Dictionary<(int x, int y), (int x, int y)[]> adjacent, (int X, int Y) start)
        {
            var visited = new HashSet<(int X, int Y)>() { start };
            var queue = new Queue<(int X, int Y)>();
            queue.Enqueue(start);
            while (queue.Count > 0)
            {
                var current = queue.Dequeue();
                visited.Add(current);
                foreach (var n in adjacent[current])
                {
                    if (queue.Contains(n) || visited.Contains(n))
                        continue;
                    queue.Enqueue(n);
                    visited.Add(n);
                }
            }

            return visited.ToArray();
        }

        public static int MinCost(int[][] grid)
        {
            var height = grid.Length;
            var width = grid[0].Length;
            var adjacent = new Dictionary<(int X, int Y), List<(int X, int Y, bool ForFree)>>();
            for (var y = 0; y < height; y++)
            {
                for (var x = 0; x < width; x++)
                {
                    var list = new List<(int X, int Y, bool ForFree)>();
                    if (x - 1 >= 0)
                        list.Add((x - 1, y, grid[y][x] == 2));
                    if (x + 1 < width)
                        list.Add((x + 1, y, grid[y][x] == 1));
                    if (y - 1 >= 0)
                        list.Add((x, y - 1, grid[y][x] == 4));
                    if (y + 1 < height)
                        list.Add((x, y + 1, grid[y][x] == 3));
                    adjacent.Add((x, y), list);
                }
            }

            var queue = new Queue<(int X, int Y, int Cost)>();
            queue.Enqueue((0, 0, 0));
            var dp = new int[height][];
            for (var y = 0; y < height; y++)
            {
                dp[y] = new int[width];
                for (var x = 0; x < width; x++)
                    dp[y][x] = int.MaxValue;
            }

            var ans = 0;
            while (queue.Count > 0)
            {
                var p = queue.Dequeue();
                var cost = p.Cost;
                if (dp[p.Y][p.X] < cost)
                    continue;
                if (p.X == width - 1 && p.Y == height)
                    ans = Math.Min(dp[p.Y][p.X], p.Cost);
                foreach (var n in adjacent[(p.X, p.Y)])
                {
                    var newCost = cost + (n.ForFree ? 0 : 1);
                    if (dp[n.Y][n.X] <= newCost)
                        continue;
                    dp[n.Y][n.X] = newCost;
                    queue.Enqueue((n.X, n.Y, newCost));
                }
            }

            return ans;
        }

        public static int AlternatingSubarray(int[] nums)
        {
            var source = new int[nums.Length];
            for (var i = 0; i < nums.Length - 1; i++)
            {
                source[i] = nums[i + 1] - nums[i];
            }

            var maxLength = 0;

            for (var i = 0; i < nums.Length - 1; i++)
            {
                while (source[i] != 1)
                {
                    i++;
                }

                var length = 1;
                var needMinus = true;
                for (var right = i + 1; right < nums.Length - 1; right++)
                {
                    var value = needMinus ? -1 : 1;
                    if (source[right] == value)
                    {
                        length++;
                        maxLength = Math.Max(length, maxLength);
                        needMinus = !needMinus;
                    }
                    else
                    {
                        i = right - 1;
                        break;
                    }
                }
            }

            return maxLength + 1;
        }

        public bool DoesValidArrayExist(int[] derived)
        {
            var xor = derived[0];
            for (var i = 1; i < derived.Length; i++)
                xor ^= derived[i];
            return xor == 0;
        }

        public static int XorAllNums(int[] nums1, int[] nums2)
        {
            var xorA = nums1[0];
            for (var i = 1; i < nums1.Length; i++)
                xorA ^= nums1[i];
            var xorB = nums2[0];
            for (var i = 1; i < nums2.Length; i++)
                xorB ^= nums2[i];
            var x = 0;
            if (nums1.Length % 2 == 1)
                x = 1;
            var y = 0;
            if (nums2.Length % 2 == 1)
                y = 1;
            return (y * xorA) ^ (x * xorB);
        }

        public static int MinimizeXor(int num1, int num2)
        {
            var bitsCount = 0;
            for (var i = 0; i < 32; i++)
                if ((num2 & (1 << i)) >> i == 1)
                    bitsCount++;
            var setBits = new HashSet<int>(32);
            for (var i = 31; i >= 0 && bitsCount > 0; i--)
            {
                if ((num1 & (1 << i)) >> i == 1)
                {
                    setBits.Add(i);
                    bitsCount--;
                }
            }

            for (var i = 0; i < 32 && bitsCount > 0; i++)
            {
                if (setBits.Contains(i))
                    continue;
                if ((num1 & (1 << i)) >> i == 0)
                {
                    bitsCount--;
                    setBits.Add(i);
                }
            }

            var x = 0;
            foreach (var power in setBits)
                x += 1 << power;
            return x;
        }

        public static int[] FindThePrefixCommonArray(int[] A, int[] B)
        {
            var prefix = new int[A.Length];
            prefix[A.Length - 1] = A.Length;
            for (var i = A.Length - 2; i >= 0; i--)
            {
                prefix[i] = prefix[i + 1] - 2;
                if (A[i + 1] == B[i + 1])
                    prefix[i]++;
                prefix[i] = Math.Max(0, prefix[i]);
            }

            return prefix;
        }

        public static int MinimumLength(string s)
        {
            var alphabet = new int[26];
            foreach (var t in s)
            {
                alphabet[t - 'a']++;
            }

            var length = s.Length;
            for (var i = 0; i < alphabet.Length; i++)
            {
                var n = alphabet[i];
                if (n < 3)
                    continue;
                length -= (n - 2 + n % 2);
            }

            return length;
        }

        public static bool CanBeValid(string s, string locked)
        {
            if (s.Length % 2 == 1)
                return false;
            var unlocked = new Stack<int>();
            var open = new Stack<int>();
            for (var i = 0; i < s.Length; i++)
            {
                if (locked[i] == '0')
                    unlocked.Push(i);
                else if (s[i] == '(')
                {
                    open.Push(i);
                }
                else if (s[i] == ')')
                {
                    if (open.Count != 0)
                        open.Pop();
                    else if (unlocked.Count != 0)
                        unlocked.Pop();
                    else
                        return false;
                }
            }

            while (open.Count != 0 && unlocked.Count != 0 && open.Peek() < unlocked.Peek())
            {
                open.Pop();
                unlocked.Pop();
            }

            return open.Count == 0 && unlocked.Count % 2 == 0;
        }

        public bool CanConstruct(string s, int k)
        {
            if (s.Length < k)
                return false;
            if (s.Length == k)
                return true;
            var alphabet = new int[26];
            foreach (var x in s)
                alphabet[x - 'a']++;
            var oddLengths = 0;
            for (var i = 0; i < alphabet.Length; i++)
            {
                if (alphabet[i] % 2 == 1)
                    oddLengths++;
            }

            if (oddLengths > k)
                return false;
            return true;
        }

        public static IList<string> WordSubsets(string[] words1, string[] words2)
        {
            var universalStrings = new List<string>(words1.Length);
            var general = new int[26];
            foreach (var word in words2)
            {
                var letters = new int[26];
                foreach (var x in word)
                    letters[x - 'a']++;
                for (var i = 0; i < 26; i++)
                    general[i] = Math.Max(letters[i], general[i]);
            }

            foreach (var word in words1)
            {
                var letters = new int[26];
                foreach (var x in word)
                    letters[x - 'a']++;
                var isSubset = true;
                for (var i = 0; i < 26; i++)
                {
                    if (letters[i] >= general[i])
                        continue;
                    isSubset = false;
                    break;
                }

                if (isSubset)
                    universalStrings.Add(word);
            }

            return universalStrings;
        }

        public static int LongestAlternatingSubarray(int[] nums, int threshold)
        {
            var l = 0;
            while (l < nums.Length && (nums[l] > threshold || nums[l] % 2 == 1))
                l++;
            var maxLength = l == nums.Length ? 0 : 1;
            for (var r = l + 1; r < nums.Length; r++)
            {
                if ((nums[r] + nums[r - 1]) % 2 == 0 || nums[r] > threshold)
                {
                    l = r;
                    while (l < nums.Length && (nums[l] > threshold || nums[l] % 2 == 1))
                        l++;
                }
                else
                    maxLength = Math.Max(maxLength, r - l + 1);
            }

            return maxLength;
        }

        public int PrefixCount1(string[] words, string pref)
        {
            var prefixCount = 0;
            foreach (var x in words)
            {
                if (x.Length < pref.Length)
                    continue;
                var starts = true;
                for (var i = 0; i < pref.Length; i++)
                    if (x[i] != pref[i])
                    {
                        starts = false;
                        break;
                    }

                if (starts)
                    prefixCount++;
            }

            return prefixCount;
        }

        public int CountPrefixSuffixPairs(string[] words)
        {
            var pairs = 0;
            for (var i = 0; i < words.Length; i++)
            for (var j = i + 1; j < words.Length; j++)
            {
                var word = words[j];
                var prefix = words[i];
                if (IsPrefixAndSuffix(word, prefix))
                    pairs++;
            }

            return pairs;
        }

        bool IsPrefixAndSuffix(string word, string prefix)
        {
            if (prefix.Length > word.Length)
                return false;
            for (var i = 0; i < prefix.Length; i++)
            {
                if (word[i] != prefix[i])
                    return false;
                if (word[word.Length - 1 - i] != prefix[prefix.Length - 1 - i])
                    return false;
            }

            return true;
        }

        public static IList<string> StringMatching1(string[] words)
        {
            var list = new List<string>(words.Length);
            Array.Sort(words, (x, y) => x.Length.CompareTo(y.Length));
            for (var i = 0; i < words.Length; i++)
            for (var j = i + 1; j < words.Length; j++)
            {
                if (words[j].Contains(words[i]))
                {
                    list.Add(words[i]);
                    break;
                }
            }

            return list;
        }

        public static int[] MinOperations(string boxes)
        {
            var result = new int[boxes.Length];
            for (var i = 0; i < boxes.Length; i++)
            {
                for (var j = 0; j < boxes.Length; j++)
                {
                    if (i == j)
                        continue;
                    if (boxes[j] == '0')
                        continue;
                    result[i] += Math.Abs(j - i);
                }
            }

            return result;
        }

        public static string ShiftingLetters(string s, int[][] shifts)
        {
            var actions = new int[s.Length];
            foreach (var x in shifts)
            {
                var start = x[0];
                var end = x[1];
                var direction = x[2] == 1;
                for (var i = start; i <= end; i++)
                    actions[i] += direction ? 1 : -1;
            }

            var sb = new StringBuilder(s);
            for (var i = 0; i < s.Length; i++)
            {
                var shift = actions[i] % 26;
                if (shift < 0) shift += 26;

                sb[i] = (char)('a' + (sb[i] - 'a' + shift) % 26);
            }

            return sb.ToString();
        }

        public static bool ValidPalindrome(string s)
        {
            var left = 0;
            var right = s.Length - 1;
            while (left < right)
            {
                if (s[left] != s[right])
                    break;
                left++;
                right--;
            }

            if (left >= right)
                return true;
            var l = left + 1;
            var r = right;
            while (l < r)
            {
                if (s[l] != s[r])
                    break;
                l++;
                r--;
            }

            if (l >= r)
                return true;
            l = left;
            r = right - 1;
            while (l < r)
            {
                if (s[l] != s[r])
                    break;
                l++;
                r--;
            }

            if (l >= r)
                return true;
            return false;
        }

        public static int CountPalindromicSubsequence(string s)
        {
            var alphabet = new int[26];
            foreach (var x in s)
                alphabet[x - 'a']++;
            var palindromes = 0;
            for (var i = 0; i < alphabet.Length; i++)
            {
                if (alphabet[i] < 2)
                    continue;
                var letter = (char)(i + 'a');
                var start = s.IndexOf(letter);
                var end = s.LastIndexOf(letter);
                var visited = new bool[26];
                for (var j = start + 1; j < end; j++)
                {
                    if (visited[s[j] - 'a'])
                        continue;
                    palindromes++;
                    visited[s[j] - 'a'] = true;
                }
            }

            return palindromes;
        }

        public static int WaysToSplitArray(int[] nums)
        {
            long leftSum = 0;
            foreach (var t in nums)
                leftSum += t;

            long rightSum = 0;
            var result = 0;
            for (var i = nums.Length - 1; i >= 0; i--)
            {
                leftSum -= nums[i];
                rightSum += nums[i];
                if (leftSum >= rightSum)
                    result++;
            }

            return result;
        }

        public int[] VowelStrings(string[] words, int[][] queries)
        {
            var vowelsCount = new int[words.Length];
            var vowels = new HashSet<char>(new[] { 'a', 'e', 'i', 'o', 'u' });
            vowelsCount[0] = vowels.Contains(words[0][0]) && vowels.Contains(words[0][^1]) ? 1 : 0;
            for (var i = 1; i < words.Length; i++)
            {
                vowelsCount[i] = vowelsCount[i - 1];
                if (vowels.Contains(words[i][0]) && vowels.Contains(words[i][^1]))
                    vowelsCount[i]++;
            }

            var result = new int[queries.Length];
            for (var i = 0; i < queries.Length; i++)
            {
                result[i] = vowelsCount[queries[i][1]];
                if (queries[i][0] > 0)
                    result[i] -= vowelsCount[queries[i][0] - 1];
            }

            return result;
        }

        // bool IsVowelString(string word)
        // {
        //     var vowels = new HashSet<char>(new[] { 'a', 'e', 'i', 'o', 'u' });
        //     return vowels.Contains(word[0]) && vowels.Contains(word[^1]);
        // }

        static int GetVowelSubstringsCount(string word)
        {
            var left = 0;
            var substrings = 0;
            for (var right = 0; right < word.Length; right++)
            {
                while (right < word.Length && !Vowels.Contains(word[right]))
                {
                    right++;
                }

                if (right == word.Length)
                    break;
                substrings++;
                while (left < right)
                {
                    if (Vowels.Contains(word[left]))
                        substrings++;
                    left++;
                }
            }

            return substrings;
        }

        public static int MaxScore(string s)
        {
            var left = new int[s.Length];
            left[0] = s[0] == '0' ? 1 : 0;
            var right = new int[s.Length];
            right[right.Length - 1] = s[right.Length - 1] == '1' ? 1 : 0;
            for (var i = 1; i < s.Length - 1; i++)
            {
                if (s[i] == '0')
                    left[i] = left[i - 1] + 1;
                else
                    left[i] = left[i - 1];
            }

            for (var i = right.Length - 2; i > 0; i--)
            {
                if (s[i] == '1')
                    right[i] = right[i + 1] + 1;
                else
                    right[i] = right[i + 1];
            }

            var max = 0;
            for (var i = 0; i < left.Length - 1; i++)
            {
                max = Math.Max(max, left[i] + right[i + 1]);
            }

            return max;
        }

        public static bool CheckDistances(string s, int[] distance)
        {
            var actualDistances = new Dictionary<char, int>();
            for (var i = 0; i < s.Length; i++)
            {
                if (!actualDistances.ContainsKey(s[i]))
                    actualDistances[s[i]] = i;
                else
                {
                    actualDistances[s[i]] = i - actualDistances[s[i]] - 1;
                }
            }

            foreach (var x in actualDistances)
            {
                if (x.Value != distance[x.Key - 'a'])
                    return false;
            }

            return true;
        }

        public static int MincostTickets(int[] days, int[] costs)
        {
            var dp = new int[366];
            dp[days[days.Length - 1]] = 1;
            for (var i = 1; i <= 365; i++)
            {
                if (days.Contains(i))
                {
                    var dayBefore = dp[i - 1] + costs[0];
                    var sevenBefore = i - 7 >= 0 ? dp[i - 7] + costs[1] : int.MaxValue;
                    var thirtyBefore = i - 30 >= 0 ? dp[i - 30] + costs[2] : int.MaxValue;
                    var minCost = Math.Min(dayBefore, Math.Min(sevenBefore, thirtyBefore));
                    dp[i] = minCost;
                }
                else
                {
                    dp[i] = dp[i - 1];
                }
            }

            return dp[days[days.Length - 1]];
        }

        public int CountGoodStrings(int minLength, int maxLength, int n, int k)
        {
            var dp = new int[maxLength + 1];
            dp[n]++;
            dp[k]++;

            for (var i = 0; i <= maxLength; i++)
            {
                if (i + n <= maxLength)
                    dp[i + n] = (dp[i + n] + dp[i]) % 1000000007;
                if (i + k <= maxLength)
                    dp[i + k] = (dp[i + k] + dp[i]) % 1000000007;
            }

            var result = 0;
            for (var i = minLength; i <= maxLength; i++)
                result = (result + dp[i]) % 1000000007;

            return result;
        }

        public static int GenerateKey(int num1, int num2, int num3)
        {
            var keys = new int[4];
            var pointer = 0;
            while (num3 > 0 || num2 > 0 || num1 > 0)
            {
                keys[pointer] = Math.Min(num1 % 10, Math.Min(num2 % 10, num3 % 10));
                pointer++;
                num1 /= 10;
                num2 /= 10;
                num3 /= 10;
            }

            var key = 0;
            var multiplier = 1;
            for (var i = 0; i < 4; i++)
            {
                key += multiplier * keys[i];
                multiplier *= 10;
            }

            return key;
        }

        static int[] GetFrequencyByPosition(string[] words, int position)
        {
            var frequency = new int[26];
            foreach (var w in words)
            {
                frequency[w[position] - 'a']++;
            }

            return frequency;
        }

        public static int NumWays(string[] words, string target)
        {
            var numWays = new long[target.Length];
            for (var p = 0; p < words[0].Length; p++)
            {
                var frequency = GetFrequencyByPosition(words, p);
                for (var i = Math.Min(p, target.Length - 1); i >= 0; i--)
                {
                    var letter = target[i];
                    var f = frequency[letter - 'a'];
                    if (f == 0)
                        continue;
                    numWays[i] += i == 0 ? f : numWays[i - 1] * f;
                    numWays[i] %= 1000000007;
                }
            }

            return (int)numWays[^1];
        }

        public static int[] MaxSumOfThreeSubarrays1(int[] nums, int k)
        {
            for (int i = 1; i < nums.Length; i++)
                nums[i] += nums[i - 1];
            var x = 0;
            var result = new int[3];
            for (var i = k; i <= nums.Length - 2 * k; i++)
            {
                var l = 0;
                var maxRight = 0;
                var r = i + k;
                var maxLeft = Math.Max(0, nums[k - 1]);
                for (var j = 0; j + k < i; j++)
                {
                    if (maxLeft >= nums[j + k] - nums[j])
                        continue;
                    maxLeft = nums[j + k] - nums[j];
                    l = j + 1;
                }

                for (var j = i + k; j <= nums.Length - k; j++)
                {
                    if (maxRight >= nums[j + k - 1] - nums[j - 1])
                        continue;
                    maxRight = nums[j + k - 1] - nums[j - 1];
                    r = j;
                }

                if (nums[i + k - 1] - nums[i - 1] + maxRight + maxLeft <= x)
                    continue;
                x = nums[i + k - 1] - nums[i - 1] + maxRight + maxLeft;
                result = new[] { l, i, r };
            }

            return result;
        }

        public static int[] MaxSumOfThreeSubarrays(int[] nums, int k)
        {
            for (var i = 1; i < nums.Length; i++)
                nums[i] += nums[i - 1];
            var x = 0;
            var result = new int[3];
            for (var i = k; i <= nums.Length - 2 * k; i++)
            {
                var maxLeft = 0;
                var left = 0;
                var maxRight = Math.Max(maxLeft, nums[k - 1]);
                var right = i + k;
                for (var j = 0; j + k < i; j++)
                {
                    if (maxLeft < nums[j + k] - nums[j])
                    {
                        maxLeft = nums[j + k] - nums[j];
                        left = j + 1;
                    }
                }

                for (var j = i + k; j <= nums.Length - k; j++)
                {
                    if (maxRight < nums[j + k - 1] - nums[j - 1])
                    {
                        maxRight = nums[j + k - 1] - nums[j - 1];
                        right = j;
                    }
                }

                if (nums[i + k - 1] - nums[i - 1] + maxRight + maxLeft > x)
                {
                    x = nums[i + k - 1] - nums[i - 1] + maxRight + maxLeft;
                    result = new[] { left, i, right };
                }
            }

            return result;
        }

        public static int MaxScoreSightseeingPair(int[] values)
        {
            var precomputed = new int[values.Length];
            precomputed[values.Length - 1] = values[values.Length - 1] - (values.Length - 1);
            for (var j = values.Length - 2; j >= 0; j--)
            {
                precomputed[j] = Math.Max(precomputed[j + 1], values[j] - j);
            }

            var max = int.MinValue;
            for (var i = 0; i < values.Length - 1; i++)
            {
                max = Math.Max(max, precomputed[i + 1] + values[i] + i);
            }

            return max;
        }

        private int targetsCount = 0;

        public int FindTargetSumWays(int[] nums, int target)
        {
            Find(nums, 0, target);
            return targetsCount;
        }

        public void Find(int[] nums, int position, int target)
        {
            if (position == nums.Length)
            {
                if (target == 0)
                    targetsCount++;
                return;
            }

            Find(nums, position + 1, target - nums[position]);
            Find(nums, position + 1, target + nums[position]);
        }

        public static IList<int> LargestValues(TreeNode root)
        {
            if (root == null)
                return Array.Empty<int>();
            GetTreeRows(root.left, root.right, 1);
            var result = new int[rowsByDepth.Keys.Count + 1];
            result[0] = root.val;
            for (var i = 1; i < result.Length; i++)
            {
                result[i] = rowsByDepth[i].Max();
            }


            return result;
        }

        static Dictionary<int, List<int>> rowsByDepth = new();

        static void GetTreeRows(TreeNode left, TreeNode right, int depth)
        {
            if (left == null && right == null)
                return;
            if (!rowsByDepth.ContainsKey(depth))
                rowsByDepth[depth] = new List<int>();
            if (left != null)
                rowsByDepth[depth].Add(left.val);
            if (right != null)
                rowsByDepth[depth].Add(right.val);
            GetTreeRows(left?.left, left?.right, depth + 1);
            GetTreeRows(right?.left, right?.right, depth + 1);
        }

        public static int MinimumDiameterAfterMerge(int[][] edges1, int[][] edges2)
        {
            if (edges1.Length == 0 || edges2.Length == 0)
                return 1;
            var adjacent1 = GetAdjacent(edges1);
            var adjacent2 = GetAdjacent(edges2);
            var (_, end1) = adjacent1.Count > 0 ? GetMaxDepth(adjacent1, adjacent1.Keys.First()) : (0, 0);
            var (maxDepth1, _) = adjacent1.Count > 0 ? GetMaxDepth(adjacent1, end1) : (0, 0);
            ;
            var (_, end2) = adjacent2.Count > 0 ? GetMaxDepth(adjacent2, adjacent2.Keys.First()) : (0, 0);
            var (maxDepth2, _) = adjacent2.Count > 0 ? GetMaxDepth(adjacent2, end2) : (0, 0);
            return (int)Math.Max(
                Math.Max(maxDepth1, maxDepth2),
                Math.Ceiling(maxDepth1 / 2.0) + Math.Ceiling(maxDepth2 / 2.0) + 1);
        }

        private static Dictionary<int, List<int>> GetAdjacent(int[][] edges)
        {
            var adjacent = new Dictionary<int, List<int>>();
            if (edges.Length == 0)
                return adjacent;
            foreach (var x in edges)
            {
                if (!adjacent.ContainsKey(x[0]))
                    adjacent.Add(x[0], new List<int>());
                adjacent[x[0]].Add(x[1]);
                if (!adjacent.ContainsKey(x[1]))
                    adjacent.Add(x[1], new List<int>());
                adjacent[x[1]].Add(x[0]);
            }

            return adjacent;
        }

        static (int Depth, int End) GetMaxDepth(Dictionary<int, List<int>> adjacent, int start)
        {
            var maxDepth = 0;
            var end = start;
            var visited = new HashSet<int>();
            var queue = new Queue<(int Node, int Depth)>();
            queue.Enqueue((start, 0));
            while (queue.Count > 0)
            {
                var (node, depth) = queue.Dequeue();
                if (!visited.Add(node))
                    continue;
                if (depth > maxDepth)
                {
                    maxDepth = depth;
                    end = node;
                }

                foreach (var x in adjacent[node])
                {
                    queue.Enqueue((x, depth + 1));
                }
            }

            return (maxDepth, end);
        }

        public int DuplicateNumbersXOR(int[] nums)
        {
            var dict = new Dictionary<int, int>(nums.Length);
            foreach (var x in nums)
            {
                if (!dict.ContainsKey(x))
                    dict.Add(x, 0);
                dict[x]++;
            }

            if (dict.All(x => x.Value != 2))
                return 0;
            return dict.Where(x => x.Value == 2)
                .Select(x => x.Key)
                .Aggregate((x, y) => x ^ y);
        }

        public int SumIndicesWithKSetBits(IList<int> nums, int k)
        {
            var sum = 0;
            for (var i = 0; i < nums.Count; i++)
            {
                var setBits = 0;
                for (var p = 0; p < 32 && 1 << p <= i; p++)
                {
                    if ((i & (1 << p)) >> p == 1)
                        setBits++;
                }

                if (setBits == k)
                    sum += nums[i];
            }

            return sum;
        }

        // public int MinimumOperations(TreeNode root)
        // {
        //     // var queue = new Queue<TreeNode>();
        //     // queue.Enqueue(root);
        //     // var visited = new HashSet<int>();
        //     // while (queue.Count > 0)
        //     // {
        //     //     var node = queue.Dequeue();
        //     //     if (node == null)
        //     //         continue;
        //     //     if (!visited.Add(node.val))
        //     //         continue;
        //     //     queue.Enqueue(node.left);
        //     //     queue.Enqueue(node.right);
        //     // }
        //     Meow1(root.left, root.right, 1);
        //     var result = 0;
        //     foreach (var level in dict)
        //     {
        //         result += SwapCountToMakeSorted(level.Value.ToArray());
        //     }
        //
        //     return result;
        // }
        //
        // Dictionary<int, List<int>> dict = new();
        // void Meow1(TreeNode left, TreeNode right, int depth)
        // {
        //     if (!dict.ContainsKey(depth))
        //         dict[depth] = new List<int>();
        //     if (left != null)
        //         dict[depth].Add(left.val);
        //     if (right != null)
        //         dict[depth].Add(right.val);
        //     Meow1(left?.left, left?.right, depth + 1);
        //     Meow1(right?.left, right?.right, depth + 1);
        // }

        static int SwapCountToMakeSorted(List<int> collection)
        {
            var swapCount = 0;
            var sorted = collection.ToArray();
            Array.Sort(sorted);
            var indexed = new Dictionary<int, int>();
            for (var i = 0; i < sorted.Length; i++)
                indexed.Add(collection[i], i);
            for (var i = 0; i < collection.Count; i++)
            {
                if (collection[i] != sorted[i])
                {
                    swapCount++;
                    collection[indexed[sorted[i]]] = collection[i];
                }
            }

            Console.WriteLine(swapCount);
            return swapCount;
        }

        public static int[] LeftmostBuildingQueries(int[] heights, int[][] queries)
        {
            var result = new int[queries.Length];
            foreach (var x in queries)
            {
                if (x[1] >= x[0])
                    continue;
                (x[0], x[1]) = (x[1], x[0]);
            }

            var dict = new Dictionary<(int X, int Y), int>();
            var queue = new PriorityQueue<(int x, int y, int index), int>();
            // for (var i = 0; i < queries.Length; i++)
            // {
            //     queue.Enqueue((queries[i][0], queries[i][1], i), queries[i][1]);
            // }

            for (var i = 0; i < queries.Length; i++)
            {
                var (a, b) = (queries[i][0], queries[i][1]);
                if (dict.ContainsKey((a, b)))
                {
                    result[i] = dict[(a, b)];
                    continue;
                }

                if (a == b || (a < b && heights[a] < heights[b]))
                {
                    result[i] = b;
                    dict.Add((a, b), result[i]);
                    continue;
                }

                var minX = Math.Max(a, b);
                var minHeight = Math.Max(heights[a], heights[b]);
                queue.Enqueue((minX, minHeight, i), -minHeight);
            }

            var sortedHeights = new List<(int Height, int Index)>(heights.Length);
            sortedHeights.AddRange(heights
                .Select((h, i) => (h, i))
                .OrderByDescending(x => x.h));
            var sortedList = new SortedList<int, int>();
            var tree = new SortedSet<int>();
            var sortedHeightIndex = 0;
            while (queue.Count > 0)
            {
                var (x, h, index) = queue.Dequeue();
                while (sortedHeightIndex < sortedHeights.Count && h < sortedHeights[sortedHeightIndex].Height)
                {
                    tree.Add(sortedHeights[sortedHeightIndex].Index);
                    sortedList.Add(sortedHeights[sortedHeightIndex].Index, sortedHeights[sortedHeightIndex].Index);
                    sortedHeightIndex++;
                }

                var value = tree.GetViewBetween(x + 1, heights.Length);
                if (value.Count == 0)
                    result[index] = -1;
                else
                    result[index] = value.Min;

                // var list = sortedList.Keys;
                // var left = x+1;
                // var right = list.Count - 1;
                // while (left < right)
                // {
                //     var middle = (left + right) / 2;
                //     if (x <= list[middle])
                //         right = middle;
                //     else 
                //         left = middle + 1;
                // }
                //
                // if (left == list.Count || list[left] <= x)
                //     result[index] = -1;
                // else
                //     result[index] = list[left];
            }

            return result;
        }

        // public int MaxKDivisibleComponents(int n, int[][] edges, int[] values, int k)
        // {
        //     foreach (var e in edges)
        //     {
        //         if (!adjacent.ContainsKey(e[0]))
        //             adjacent.Add(e[0], new List<int>());
        //         adjacent[e[0]].Add(e[1]);
        //         if (!adjacent.ContainsKey(e[1]))
        //             adjacent.Add(e[1], new List<int>());
        //         adjacent[e[1]].Add(e[0]);
        //     }
        //
        //     Dfs(0, values, k);
        //     return count;
        // }

        private int count = 0;
        HashSet<int> visitedGlobal = new();
        // Dictionary<int, List<int>> adjacent = new();

        // int Dfs(int i, int[] values, int k)
        // {
        //     visitedGlobal.Add(i);
        //     var sum = values[i];
        //     if (adjacent.ContainsKey(sum))
        //         foreach (var j in adjacent[i])
        //         {
        //             if (visitedGlobal.Contains(j))
        //                 continue;
        //             visitedGlobal.Add(j);
        //             sum += Dfs(j, values, k);
        //             sum %= k;
        //         }
        //
        //     if (sum % k == 0)
        //     {
        //         count++;
        //         return 0;
        //     }
        //
        //     return sum;
        // }

        public TreeNode ReverseOddLevels(TreeNode root)
        {
            Meow(root.left, root.right, 2);
            return root;
        }

        void Meow(TreeNode left, TreeNode right, int depth)
        {
            if (left == null || right == null)
                return;
            if (depth % 2 != 0)
            {
                Meow(left.left, left.right, depth + 1);
                Meow(right.left, right.right, depth + 1);
            }

            var tmp = left.val;
            left.val = right.val;
            right.val = tmp;
            Meow(left.left, left.right, depth + 1);
            Meow(right.left, right.right, depth + 1);
        }

        public static int CountSegments(string s)
        {
            if (s.Length == 0)
                return 0;
            var pointer = 0;
            var count = 0;
            while (pointer < s.Length && s[pointer] == ' ')
                pointer++;

            for (var i = pointer; i < s.Length; i++)
            {
                if (s[i] != ' ')
                    continue;
                if (i > 0 && s[i - 1] == ' ')
                    continue;
                count++;
            }

            return s[s.Length - 1] != ' ' ? count + 1 : count;
        }

        public static int ThirdMax(int[] nums)
        {
            var visited = new HashSet<int>(3);
            Array.Sort(nums);
            var count = 0;
            for (var i = nums.Length - 1; i >= 0; i--)
            {
                if (visited.Contains(nums[i]))
                    continue;
                visited.Add(nums[i]);
                count++;
                if (count == 3)
                    return nums[i];
            }

            return nums[^1];
        }

        public static int MaxChunksToSorted(int[] arr)
        {
            var chunks = 0;
            var sum = 0;
            var realSum = 0;
            for (var i = 0; i < arr.Length; i++)
            {
                sum += i;
                realSum += arr[i];
                if (realSum == sum)
                    chunks++;
            }

            return chunks;
        }

        public int[] FinalPrices(int[] prices)
        {
            for (var i = 0; i < prices.Length; i++)
            {
                for (var j = i + 1; j < prices.Length; j++)
                    if (prices[j] <= prices[i])
                    {
                        prices[i] -= prices[j];
                        break;
                    }
            }

            return prices;
        }

        public static string RepeatLimitedString(string s, int repeatLimit)
        {
            var dict = new int[26];
            foreach (var x in s)
                dict[x - 'a']++;
            var repeatLength = 0;
            var sb = new StringBuilder(s.Length);
            for (var i = 0; i < s.Length; i++)
            {
                var currentChar = GetMaxIndex(dict);
                if (repeatLength == repeatLimit)
                {
                    repeatLength = 1;
                    if (currentChar == sb[sb.Length - 1] - 'a')
                        currentChar = GetMaxIndex(dict, sb[sb.Length - 1] - 'a' - 1);
                    if (currentChar == -1)
                        return sb.ToString();
                }

                if (sb.Length > 0 && currentChar == sb[sb.Length - 1] - 'a')
                {
                    repeatLength++;
                }
                else
                {
                    repeatLength = 1;
                }

                dict[currentChar]--;
                sb.Append((char)(currentChar + 'a'));
            }

            return sb.ToString();
        }

        static int GetMaxIndex(int[] chars, int start = 25)
        {
            for (var i = start; i >= 0; i--)
                if (chars[i] != 0)
                    return i;
            return -1;
        }

        public int[] GetFinalState(int[] nums, int k, int multiplier)
        {
            for (var i = 0; i < k; i++)
            {
                var min = int.MaxValue;
                var index = -1;
                for (var j = 0; j < k; j++)
                {
                    if (nums[i] < min)
                    {
                        min = nums[i];
                        index = i;
                    }
                }

                nums[index] *= multiplier;
            }


            return nums;
        }

        public static double MaxAverageRatio(int[][] classes, int extraStudents)
        {
            var n = classes.Length;
            var queue = new PriorityQueue<(int X, int Y), double>();
            for (var i = 0; i < n; i++)
            {
                var currentRate = Rate(classes[i][0], classes[i][1]);
                var newRate = Rate(classes[i][0] + 1, classes[i][1] + 1);
                queue.Enqueue((classes[i][0], classes[i][1]), currentRate - newRate);
            }

            while (extraStudents > 0)
            {
                var current = queue.Dequeue();
                var currentRate = Rate(current.X, current.Y);
                var newRate = Rate(current.X + 1, current.Y + 1);
                queue.Enqueue((current.X + 1, current.Y + 1), currentRate - newRate);
                extraStudents--;
            }

            var rate = 0.0;
            while (queue.Count > 0)
            {
                var current = queue.Dequeue();
                rate += Rate(current.X, current.Y);
            }

            return rate / n;
        }

        static double Rate(int X, int Y)
        {
            return (double)X / Y;
        }

        public static long ContinuousSubarrays(int[] nums)
        {
            var sortedList = new SortedList<int, int>();
            var i = 0;
            var j = 0;
            long count = 0;
            while (i <= j && j < nums.Length)
            {
                sortedList[nums[j]] = sortedList.GetValueOrDefault(nums[j], 0) + 1;
                while (Math.Abs(sortedList.Keys[0] - sortedList.Keys[^1]) > 2)
                {
                    sortedList[nums[i]]--;
                    if (sortedList[nums[i]] == 0)
                        sortedList.Remove(nums[i]);
                    i++;
                }

                count += j - i + 1;
                j++;
            }

            return count;
        }

        void Meow(int[] nums, int length)
        {
            var visited = new HashSet<string>();
        }

        public static long FindScore(int[] nums)
        {
            var marked = new HashSet<int>();
            var queue = new PriorityQueue<(int Value, int Index), (int Value, int Index)>();
            for (var i = 0; i < nums.Length; i++)
            {
                queue.Enqueue((nums[i], i), (nums[i], i));
            }

            long score = 0;
            while (queue.Count > 0)
            {
                var (value, index) = queue.Dequeue();
                if (marked.Contains(index))
                    continue;
                marked.Add(index - 1);
                marked.Add(index);
                marked.Add(index + 1);
                score += value;
            }

            return score;
        }

        public long PickGifts1(int[] gifts, int k)
        {
            var queue = new PriorityQueue<int, int>();
            foreach (var x in gifts)
            {
                queue.Enqueue(x, -x);
            }

            for (var i = 0; i < k; i++)
            {
                var x = (int)Math.Sqrt(queue.Dequeue());
                queue.Enqueue(x, -x);
            }

            var sum = 0;
            while (queue.Count > 0)
            {
                sum += queue.Dequeue();
            }

            return sum;
        }

        public static int MaximumBeauty(int[] nums, int k)
        {
            Array.Sort(nums);
            var pointer = 0;
            var max = 1;
            while (pointer < nums.Length)
            {
                var next = pointer + 1;
                while (next < nums.Length && nums[next] - nums[pointer] <= 2 * k)
                {
                    next++;
                }

                max = Math.Max(max, next - pointer);
                if (next == nums.Length)
                    break;
                while (pointer <= next && nums[next] - nums[pointer] > 2 * k)
                {
                    pointer++;
                }
            }

            return max;
        }

        public bool IsBalanced(string nums)
        {
            var even = 0;
            var odd = 0;
            for (var i = 0; i < nums.Length; i++)
            {
                if (i % 2 == 0)
                    even += nums[i] - '0';
                else
                    odd += nums[i] - '0';
            }

            return even == odd;
        }

        public int MinElement(int[] nums)
        {
            var min = int.MaxValue;
            for (var i = 0; i < nums.Length; i++)
            {
                var sum = 0;
                var x = nums[i];
                while (x > 0)
                {
                    sum += x % 10;
                    x /= 10;
                }

                min = Math.Min(min, sum);
            }

            return min;
        }

        public int MaximumLength(string s)
        {
            var maxLength = 0;
            for (var c = 0; c < 26; c++)
            {
                if (!s.Contains((char)(c + 'a')))
                    continue;
                for (var length = 1; length < 50; length++)
                {
                    var specialsWithLength = Specials(s, (char)(c + 'a'), length);
                    if (specialsWithLength < 3)
                        break;
                    maxLength = Math.Max(maxLength, length);
                }
            }

            return maxLength == 0 ? -1 : maxLength;
        }

        int Specials(string line, char x, int length)
        {
            var substring = new string(x, length);
            var start = line.IndexOf(substring);
            if (start == -1)
                return 0;
            var count = 0;
            while (start != -1)
            {
                count++;
                start = line.IndexOf(substring, start + 1);
            }

            return count;
        }

        // public static bool[] IsArraySpecial(int[] nums, int[][] queries)
        // {
        //     var solution = new bool[queries.Length];
        //     var previous = nums[0];
        //     nums[0] = 0;
        //     for (var i = 1; i < nums.Length; i++)
        //     {
        //         if ((previous + nums[i]) % 2 == 0)
        //         {
        //             nums[i] = 0;
        //             continue;
        //         }
        //
        //         previous = nums[i];
        //         nums[i] = nums[i - 1] + 1;
        //     }
        //
        //     for (var i = 0; i < queries.Length; i++)
        //     {
        //         solution[i] = nums[b] >= b - a;
        //     }
        //
        //     return solution;
        // }

        public int MaxTwoEvents(int[][] events)
        {
            events = events.OrderBy(x => x[0]).ToArray();
            var maxSuffix = new int[events.Length + 1];
            for (var i = events.Length - 1; i >= 0; i--)
                maxSuffix[i] = Math.Max(maxSuffix[i + 1], events[i][2]);
            var max = 0;
            for (var i = 0; i < events.Length; i++)
            {
                var index = Search(events, events[i][1]);
                max = Math.Max(max, events[i][2] + maxSuffix[index]);
            }

            return max;
        }

        int Search(int[][] events, int endTime)
        {
            var left = 0;
            var right = events.Length;
            while (left < right)
            {
                var m = left + (right - left) / 2;
                if (events[m][0] > endTime)
                    right = m;
                else
                    left = m;
            }

            return right;
        }

        public static int MinimumSize(int[] nums, int maxOperations)
        {
            var left = 1;
            var right = nums.Max();
            while (left < right)
            {
                var count = 0;
                var middle = left + (right - left) / 2;
                foreach (var x in nums)
                {
                    count += (x - 1) / middle;
                }

                if (count > maxOperations)
                    left = middle + 1;
                else
                    right = middle;
            }

            return left;
        }

        public int MaxCount(int[] banned, int n, int maxSum)
        {
            var excluded = new HashSet<int>(banned);
            var sum = 0;
            var count = 0;
            for (var i = 1; i <= n; i++)
            {
                if (excluded.Contains(i))
                    continue;
                if (sum + i > maxSum)
                    break;
                sum += i;
                count++;
            }

            return count;
        }

        public static bool CanChange(string source, string target)
        {
            var sourceLefts = 0;
            var sourceRights = 0;
            var targetLefts = 0;
            var targetRights = 0;
            for (var i = 0; i < source.Length; i++)
            {
                if (source[i] == 'L')
                    sourceLefts++;
                else if (source[i] == 'R')
                    sourceRights++;
                if (target[i] == 'L')
                    targetLefts++;
                else if (target[i] == 'R')
                    targetRights++;
            }

            if (sourceLefts != targetLefts || sourceRights != targetRights)
                return false;

            var sourceStack = new Stack<(char Value, int Index)>();
            var targetStack = new Stack<(char Value, int Index)>();
            for (var i = 0; i < source.Length; i++)
            {
                if (source[i] != '_')
                    sourceStack.Push((source[i], i));
                if (target[i] != '_')
                    targetStack.Push((target[i], i));
            }

            while (sourceStack.Count != 0)
            {
                var s = sourceStack.Pop();
                var t = targetStack.Pop();
                if (s.Value != t.Value)
                    return false;
                if (s.Value == 'R' && t.Index < s.Index)
                    return false;
                if (s.Value == 'L' && t.Index > s.Index)
                    return false;
            }

            return true;
        }

        public static bool CanMakeSubsequence(string s, string t)
        {
            var alphabet = Enumerable.Range(0, 26)
                .Select(x => (char)('a' + x))
                .ToArray();
            var p = 0;
            for (var i = 0; i < s.Length; i++)
            {
                var indexOfNext = (s[i] - 'a' + 1) % 26;
                // if (s[i] == t[p])
                if (s[i] == t[p] || t[p] == alphabet[indexOfNext])
                {
                    p++;
                    if (p == t.Length)
                        return true;
                }
            }

            return false;
        }

        public static bool RepeatedSubstringPattern(string s)
        {
            for (var i = 1; i < s.Length / 2 + 1; i++)
            {
                if (s.Length % i != 0)
                    continue;
                var patternFound = true;
                for (var j = 0; j < s.Length - i; j++)
                {
                    if (s[j] != s[j + i])
                    {
                        patternFound = false;
                        break;
                    }
                }

                if (patternFound)
                    return true;
            }

            return false;
        }

        public bool CheckIfExist(int[] arr)
        {
            var visited = new HashSet<int>(arr.Length);
            foreach (var x in arr)
            {
                if (visited.Contains(2 * x) || (x % 2 == 0 && visited.Contains(x / 2)))
                    return true;
                visited.Add(x);
            }

            return false;
        }

        public static int[][] ValidArrangement(int[][] pairs)
        {
            //todo или всех пар равное количество, или есть два числа 
            var adjacent = new Dictionary<int, Stack<int>>();
            var inDegree = new Dictionary<int, int>();
            var outDegree = new Dictionary<int, int>();
            foreach (var pair in pairs)
            {
                if (!inDegree.ContainsKey(pair[0]))
                    inDegree.Add(pair[0], 0);
                inDegree[pair[0]]++;
                if (!outDegree.ContainsKey(pair[1]))
                    outDegree.Add(pair[1], 0);
                outDegree[pair[1]]++;
                if (!adjacent.ContainsKey(pair[0]))
                    adjacent.Add(pair[0], new Stack<int>());
                adjacent[pair[0]].Push(pair[1]);
            }

            var result = new List<int[]>();
            var start = pairs[0][0];
            foreach (var x in adjacent.Keys)
            {
                if (outDegree.GetValueOrDefault(x, 0) - inDegree.GetValueOrDefault(x, 0) < 0)
                {
                    start = x;
                    break;
                }
            }

            var stack = new Stack<int>();
            stack.Push(start);
            while (stack.Count > 0)
            {
                var x = stack.Peek();
                if (adjacent.ContainsKey(x) && adjacent[x].Count > 0)
                {
                    stack.Push(adjacent[x].Pop());
                }
                else
                {
                    stack.Pop();
                    if (stack.Count > 0)
                        result.Add(new[] { stack.Peek(), x });
                }
            }

            result.Reverse();
            return result.ToArray();
        }

        public bool QueryString(string s, int n)
        {
            var firstOne = s.IndexOf('1');
            if (firstOne == -1)
                return false;
            s = s.Substring(firstOne);
            for (var i = 1; i <= n; i++)
                if (!s.Contains(Convert.ToString(i, 2)))
                    return false;
            return true;
        }

        public static int MinimumTime(int[][] grid)
        {
            if (grid[0][1] > 1 && grid[1][0] > 1)
            {
                return -1;
            }

            var height = grid.Length;
            var width = grid[0].Length;
            var visited = new HashSet<(int X, int Y)>();
            var queue = new PriorityQueue<(int X, int Y, int Time), int>();
            queue.Enqueue((0, 0, 0), 0);
            var neighbours = new (int dx, int dy)[] { (0, 1), (1, 0), (0, -1), (-1, 0) };
            while (queue.Count > 0)
            {
                var (x, y, time) = queue.Dequeue();
                if (visited.Contains((x, y)))
                    continue;
                visited.Add((x, y));
                if (x == width - 1 && y == height - 1)
                    return time;
                foreach (var (dx, dy) in neighbours)
                {
                    var xi = x + dx;
                    var yi = y + dy;
                    if (xi < 0 || xi >= width || yi < 0 || yi >= height)
                        continue;
                    var newTime = time + 1;
                    if (grid[yi][xi] > newTime)
                    {
                        newTime += grid[yi][xi] - newTime + (grid[yi][xi] - newTime) % 2;
                    }

                    queue.Enqueue((xi, yi, newTime), newTime);
                }
            }

            return -1;
        }

        public static int MinimumObstacles(int[][] grid)
        {
            var adjacent = new Dictionary<(int X, int Y), List<(int X, int Y)>>();
            var height = grid.Length;
            var width = grid[0].Length;
            for (var y = 0; y < height; y++)
            for (var x = 0; x < width; x++)
            {
                var neighbors = new List<(int X, int Y)>();
                if (y - 1 >= 0)
                    neighbors.Add((x, y - 1));
                if (y + 1 < height)
                    neighbors.Add((x, y + 1));
                if (x - 1 >= 0)
                    neighbors.Add((x - 1, y));
                if (x + 1 < width)
                    neighbors.Add((x + 1, y));
                adjacent.Add((x, y), neighbors);
            }

            var dist = new int[height, width];
            for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
                dist[i, j] = int.MaxValue;
            dist[0, 0] = 0;
            var visited = new HashSet<(int X, int Y)>();
            var priorityQueue = new LinkedList<(int X, int Y)>();
            priorityQueue.AddFirst((0, 0));
            while (priorityQueue.Count > 0)
            {
                var current = priorityQueue.First.Value;
                priorityQueue.RemoveFirst();
                if (visited.Contains((current.X, current.Y)))
                    continue;
                visited.Add((current.X, current.Y));
                foreach (var x in adjacent[(current.X, current.Y)])
                {
                    var d = dist[current.Y, current.X] + grid[x.Y][x.X];
                    if (d > dist[x.Y, x.X])
                        continue;
                    dist[x.Y, x.X] = d;
                    if (grid[x.Y][x.X] == 1)
                        priorityQueue.AddLast((x));
                    else
                        priorityQueue.AddFirst((x));
                }
            }

            return dist[height - 1, width - 1];
        }

        public static int[] ShortestDistanceAfterQueries(int n, int[][] queries)
        {
            var result = new int[queries.Length];
            var adjacent = new Dictionary<int, List<int>>();
            for (var x = 1; x < n; x++)
            {
                if (!adjacent.ContainsKey(x))
                    adjacent.Add(x, new List<int>());
                if (!adjacent.ContainsKey(x - 1))
                    adjacent.Add(x - 1, new List<int>());
                adjacent[x - 1].Add(x);
            }

            var pointer = 0;
            foreach (var road in queries)
            {
                if (road[0] < road[1])
                    adjacent[road[0]].Add(road[1]);
                else
                    adjacent[road[1]].Add(road[0]);
                var visited = new HashSet<int>();
                var queue = new Queue<(int Length, int Point)>();
                queue.Enqueue((0, 0));
                while (queue.Count > 0)
                {
                    var current = queue.Dequeue();
                    if (visited.Contains(current.Point))
                        continue;
                    visited.Add(current.Point);
                    if (current.Point == n - 1)
                    {
                        result[pointer++] = current.Length;
                        break;
                    }

                    foreach (var x in adjacent[current.Point])
                        queue.Enqueue((current.Length + 1, x));
                }
            }

            return result;
        }

        public int FindChampion(int n, int[][] edges)
        {
            if (edges.Length == 0)
                return n == 1 ? 0 : -1;
            var stronger = edges.Select(x => x[0]).ToHashSet();
            var weaker = edges.Select(x => x[1]).ToHashSet();
            if (stronger.Concat(weaker).ToHashSet().Count != n)
                return -1;
            var potentialWinners = stronger.Except(weaker).ToHashSet();
            return potentialWinners.Count == 1 ? potentialWinners.Single() : -1;
        }

        public static int SlidingPuzzle(int[][] board)
        {
            var available = new[]
            {
                new[] { 1, 3 },
                new[] { 0, 2, 4 },
                new[] { 1, 5 },
                new[] { 0, 4 },
                new[] { 1, 3, 5 },
                new[] { 2, 4 },
            };
            var sb = new StringBuilder(6);
            foreach (var line in board)
            foreach (var x in line)
                sb.Append(x);
            var visited = new HashSet<string>();
            var queue = new Queue<(string State, int Count)>();
            queue.Enqueue((sb.ToString(), 0));
            while (queue.Count != 0)
            {
                var state = queue.Dequeue();
                if (state.State == "123450")
                    return state.Count;
                if (visited.Contains(state.State))
                    continue;
                visited.Add(state.State);
                var index = state.State.IndexOf('0');
                foreach (var zeroPosition in available[index])
                {
                    var buffer = state.State.ToCharArray();
                    (buffer[zeroPosition], buffer[index]) = (buffer[index], buffer[zeroPosition]);
                    queue.Enqueue((new string(buffer), state.Count + 1));
                }
            }

            return -1;
        }

        public static long MaxMatrixSum(int[][] matrix)
        {
            var lessThanZero = matrix.SelectMany(x => x)
                .Where(x => x <= 0)
                .ToArray();
            if (lessThanZero.Length % 2 == 0)
                return matrix.SelectMany(x => x).Sum(x => Math.Abs((long)x));
            var max = Math.Abs(lessThanZero.OrderBy(x => x).Last());
            var min = matrix.SelectMany(x => x).Where(x => x > 0).OrderBy(x => x)
                .ToArray();
            if (min.Length == 0)
                return matrix.SelectMany(x => x).Sum(Math.Abs) - 2 * max;
            var sum = matrix.SelectMany(x => x).Sum(x => Math.Abs((long)x)) - 2 * Math.Min(max, min.First());
            return sum;
        }

        public static char[][] RotateTheBox(char[][] box)
        {
            var height = box.Length;
            var width = box[0].Length;
            var transposed = new char[width][];
            var obstacle = '*';
            var stone = '#';
            var empty = '.';
            for (var x = 0; x < width; x++)
                transposed[x] = new char[height];
            for (var x = 0; x < width; x++)
            {
                for (var y = 0; y < height; y++)
                    transposed[x][height - 1 - y] = box[y][x];
            }

            for (var y = width - 1; y >= 0; y--)
            {
                for (var x = 0; x < height; x++)
                {
                    if (transposed[y][x] != stone)
                        continue;
                    var y1 = y;
                    while (y1 + 1 < width && transposed[y1 + 1][x] == empty)
                    {
                        transposed[y1][x] = empty;
                        transposed[y1 + 1][x] = stone;
                        y1++;
                    }
                }
            }

            return transposed;
        }

        public static long CountSubstrings(string s, char c)
        {
            var count = s.Count(t => t == c);
            return (long)count * (count - 1) / 2 + count;
        }

        public int[] MaxSubsequence(int[] nums, int k)
        {
            return nums
                .Select((x, i) => (x, i))
                .OrderBy(x => x.x)
                .Skip(nums.Length - k)
                .OrderBy(x => x.i)
                .Select(x => x.x)
                .ToArray();
        }

        static long Factorial(int n)
        {
            if (n == 0)
                return 1;
            if (n < 3)
                return n;
            return n * Factorial(n - 1);
        }

        public static int MaxEqualRowsAfterFlips(int[][] matrix)
        {
            var patterns = new Dictionary<string, int>();
            foreach (var row in matrix)
            {
                var buffer = new StringBuilder(matrix[0].Length);
                foreach (var x in row)
                    buffer.Append(row[0] == 0 ? x : x ^ 1);
                var pattern = buffer.ToString();
                patterns.TryAdd(pattern, 0);
                patterns[pattern]++;
            }

            return patterns.Values.Max();
        }

        public static int CountUnguarded(int m, int n, int[][] guards, int[][] walls)
        {
            var unoccupied = 0;
            var grid = new int[m][];
            for (var y = 0; y < m; y++)
                grid[y] = new int[n];
            foreach (var g in guards)
                grid[g[0]][g[1]] = 1;
            foreach (var w in walls)
                grid[w[0]][w[1]] = 2;
            for (var y = 0; y < m; y++)
            {
                for (var x = 0; x < n; x++)
                {
                    if (y == 2 && x == 3)
                    {
                    }

                    if (grid[y][x] == 0 || grid[y][x] == 2 || grid[y][x] == 3)
                        continue;
                    for (var xi = x + 1; xi < n; xi++)
                    {
                        if (grid[y][xi] == 1 || grid[y][xi] == 2)
                            break;
                        grid[y][xi] = 3;
                    }

                    for (var xi = x - 1; xi >= 0; xi--)
                    {
                        if (grid[y][xi] == 1 || grid[y][xi] == 2)
                            break;
                        grid[y][xi] = 3;
                    }

                    for (var yi = y + 1; yi < m; yi++)
                    {
                        if (grid[yi][x] == 1 || grid[yi][x] == 2)
                            break;
                        grid[yi][x] = 3;
                    }

                    for (var yi = y - 1; yi >= 0; yi--)
                    {
                        if (grid[yi][x] == 1 || grid[yi][x] == 2)
                            break;
                        grid[yi][x] = 3;
                    }
                }
            }

            for (var y = 0; y < m; y++)
            for (var x = 0; x < n; x++)
                if (grid[y][x] == 0)
                    unoccupied++;
            return unoccupied;
        }

        public static int TakeCharacters(string s, int k)
        {
            if (k == 0)
                return 0;
            var dict = new int[3];
            foreach (var x in s)
                dict[x - 'a']++;
            if (dict.Any(x => x < k))
                return -1;
            var left = new int[3];
            var minLength = int.MaxValue;
            var right = new int[3];

            for (var r = 1; r <= s.Length; r++)
            {
                right[s[s.Length - r] - 'a']++;
                if (right[0] >= k && right[1] >= k && right[2] >= k)
                {
                    minLength = r;
                    break;
                }
            }

            var rightLength = minLength;
            for (var leftLength = 1; leftLength < s.Length; leftLength++)
            {
                if (rightLength == 0)
                    break;
                var deleted = s[leftLength - 1];
                var added = s[s.Length - rightLength];
                left[deleted - 'a']++;
                if (added == deleted) //todo значит, для него добавляли, убавим
                {
                    right[s[s.Length - rightLength] - 'a']--;
                    rightLength--;
                    while (rightLength > 0 &&
                           right[s[s.Length - rightLength] - 'a'] + left[s[s.Length - rightLength] - 'a'] > k)
                    {
                        right[s[s.Length - rightLength] - 'a']--;
                        rightLength--;
                    }

                    minLength = Math.Min(minLength, leftLength + rightLength);
                }
            }

            return minLength;
        }

        public static long MaximumSubarraySum(int[] nums, int k)
        {
            var maxSum = 0L;
            var hashSet = new HashSet<int>(k);
            var sum = 0L;
            var left = 0;
            for (var i = 0; i < nums.Length; i++)
            {
                if (hashSet.Contains(nums[i]))
                {
                    //todo slide while x != nums[i] and sum -= x
                    while (nums[left] != nums[i])
                    {
                        hashSet.Remove(nums[left]);
                        sum -= nums[left];
                        left++;
                    }

                    // sum -= nums[left];
                    left++;
                }
                else
                {
                    hashSet.Add(nums[i]);
                    sum += nums[i];
                    if (i - left + 1 == k)
                    {
                        maxSum = Math.Max(maxSum, sum);
                        hashSet.Remove(nums[left]);
                        sum -= nums[left];
                        left++;
                    }
                }
            }

            return maxSum;
        }

        public int ShortestSubarray(int[] nums, int k)
        {
            var prefix = new int[nums.Length + 1];
            for (var i = 0; i < nums.Length; i++)
            {
                prefix[i + 1] = prefix[i] + nums[i];
            }

            var minLength = int.MaxValue;
            var deque = new LinkedList<int>();
            for (var i = 0; i < prefix.Length; i++)
            {
                while (deque.Count > 0 && prefix[i] - prefix[deque.First!.Value] >= k)
                {
                    minLength = Math.Min(minLength, i - deque.First!.Value);
                    deque.RemoveFirst();
                }

                while (deque.Count > 0 && prefix[i] - prefix[deque.Last!.Value] <= 0)
                {
                    deque.RemoveLast();
                }

                deque.AddLast(i);
            }

            return minLength == int.MaxValue ? -1 : minLength;
        }

        public int FindLengthOfShortestSubarray(int[] nums)
        {
            var right = -1;
            for (var i = 0; i < nums.Length - 1; i++)
                if (nums[i] > nums[i + 1])
                {
                    right = i;
                    break;
                }

            var left = -1;
            for (var i = nums.Length - 1; i > 0; i--)
                if (nums[i - 1] > nums[i])
                {
                    left = i;
                    break;
                }

            Console.WriteLine(left);
            Console.WriteLine(right);
            if (left == -1 && right != -1)
                return 0;
            var minLength = Math.Min(right + 1, nums.Length - left - 1);
            for (var i = 0; i <= left; i++)
            {
                var length = 0;
                for (var j = right; j < nums.Length; j++)
                {
                    if (nums[j] >= nums[i])
                    {
                        length = j - i + 1;
                        minLength = Math.Min(minLength, length);
                        break;
                    }
                }
            }

            return minLength;
        }

        public static long CountFairPairs(int[] nums, int lower, int upper)
        {
            Array.Sort(nums);
            return LessCount(nums, upper + 1) - LessCount(nums, lower);
        }

        public static int NumDifferentIntegers(string word)
        {
            var visited = new HashSet<string>();
            for (var i = 0; i < word.Length; i++)
            {
                var sb = new StringBuilder(32);
                if (char.IsDigit(word[i]))
                {
                    while (i < word.Length && char.IsDigit(word[i]))
                    {
                        if (sb.Length == 1 && sb[0] == '0')
                        {
                            sb = new StringBuilder();
                        }

                        sb.Append(word[i]);
                        i++;
                    }

                    visited.Add(sb.ToString());
                }
            }

            return visited.Count;
        }

        public static long LessCount(int[] nums, int value)
        {
            var left = 0;
            var right = nums.Length - 1;
            long count = 0;
            while (left < right)
            {
                if (nums[left] + nums[right] < value)
                {
                    count += right - left;
                    left++;
                }
                else
                {
                    right--;
                }
            }

            return count;
        }

        public static int[] MaximumBeauty(int[][] items, int[] queries)
        {
            var maxBeauty = new Dictionary<int, int>();
            for (var i = 0; i < items.Length; i++)
            {
                if (!maxBeauty.ContainsKey(items[i][0]))
                    maxBeauty.Add(items[i][0], 0);
                maxBeauty[items[i][0]] = Math.Max(maxBeauty[items[i][0]], items[i][1]);
            }

            var beauty = maxBeauty
                .Select(x => (x.Key, x.Value))
                .OrderBy(x => x.Key)
                .ToArray();
            var localMax = beauty[0].Value;
            for (var i = 1; i < beauty.Length; i++)
            {
                if (beauty[i].Value > localMax)
                    localMax = beauty[i].Value;
                else
                    beauty[i].Value = localMax;
            }

            var sortedQueries = queries
                .Select((x, i) => (x, i))
                .OrderBy(x => x.x)
                .ToArray();
            for (var i = 0; i < sortedQueries.Length; i++)
            {
                var max = i > 0 ? sortedQueries[i - 1].x : 0;
                var index = GetLeftBorderIndex(beauty, sortedQueries[i].x, -1, beauty.Length);
                var value = index != -1 ? beauty[index].Value : 0;
                max = Math.Max(max, value);
                sortedQueries[i].x = max;
            }

            return sortedQueries.OrderBy(x => x.i)
                .Select(x => x.x)
                .ToArray();
        }

        public static int GetLeftBorderIndex((int, int)[] phrases, int value, int left, int right)
        {
            if (left == right - 1)
                return left;
            var m = left + (right - left) / 2;
            var phrase = phrases[m].Item1;
            if (phrase > value)
                return GetLeftBorderIndex(phrases, value, left, m);
            return GetLeftBorderIndex(phrases, value, m, right);
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

        public static string ReversePrefix(string word, char ch)
        {
            var nonReversedIndex = -1;
            for (var i = 0; i < word.Length; i++)
            {
                if (word[i] == ch)
                {
                    nonReversedIndex = i;
                    break;
                }
            }

            if (nonReversedIndex == -1)
                return word;
            var buffer = new char[word.Length];
            for (var i = 0; i < word.Length; i++)
            {
                if (i <= nonReversedIndex)
                {
                    buffer[nonReversedIndex - i] = word[i];
                }
                else
                    buffer[i] = word[i];
            }

            return new string(buffer);
        }

        public bool IsSumEqual(string firstWord, string secondWord, string targetWord)
        {
            var dict = "abcdefghij"
                .Select((x, i) => (x, i))
                .ToDictionary(x => x.x, x => x.i);
            return GetSum(firstWord, dict) + GetSum(secondWord, dict) == GetSum(targetWord, dict);
        }

        private int GetSum(string word, Dictionary<char, int> dict)
        {
            var power = 1;
            var sum = 0;
            for (var i = word.Length - 1; i >= 0; i--)
            {
                sum += power * dict[word[i]];
                power *= 10;
            }

            return sum;
        }

        public int MinimumOperations(int[] nums)
        {
            return nums.Count(x => x % 3 != 0);
        }

        public static int LargestCombination(int[] candidates)
        {
            var binary = candidates
                .Select(x => new
                {
                    Original = x, Bin = Convert.ToString(x, 2),
                    Huj = new string(Convert.ToString(x, 2).Reverse().ToArray())
                })
                .OrderByDescending(x => x.Bin)
                .ToArray();
            var value = binary[0].Original;
            for (int i = 0; i < binary.Length - 1; i++)
            {
                if ((value & binary[i].Original) == 0)
                {
                    return i + 1;
                }
            }

            return candidates.Length;
        }

        public static string CompressedString(string word)
        {
            var builder = new StringBuilder(word.Length);
            var prefixLength = 0;
            var prefixChar = '0';
            for (var i = 0; i < word.Length; i++)
            {
                if (i == 0)
                {
                    prefixChar = word[i];
                    prefixLength++;
                    continue;
                }

                if (word[i] == prefixChar && prefixLength != 9)
                {
                    prefixLength++;
                    continue;
                }

                builder.Append(prefixLength);
                builder.Append(prefixChar);
                prefixChar = word[i];
                prefixLength = 1;
            }

            builder.Append(prefixLength);
            builder.Append(prefixChar);
            return builder.ToString();
        }

        public IList<string> ValidStrings(int n)
        {
            var initialString = new string('1', n);
            var result = new List<string> { initialString };
            MakeString(new StringBuilder(initialString), 0, result);
            return result;
        }

        void MakeString(StringBuilder sb, int index, List<string> result)
        {
            if (index >= sb.Length)
                return;
            sb[index] = '0';
            result.Add(sb.ToString());
            MakeString(sb, index + 2, result);
            sb[index] = '1';
            MakeString(sb, index + 1, result);
        }

        public string MakeFancyString(string line)
        {
            var fancy = new StringBuilder(line.Length);
            for (var i = 0; i < 2 && i < line.Length; i++)
            {
                fancy.Append(line[i]);
            }

            for (var i = 2; i < line.Length; i++)
            {
                if (line[i - 2] == line[i] && line[i - 1] == line[i])
                    continue;
                fancy.Append(line[i]);
            }

            return fancy.ToString();
        }

        public string ThousandSeparator(int n)
        {
            var buffer = new StringBuilder(20);
            var pointer = 0;
            while (n > 0)
            {
                buffer.Append(n % 10);
                n /= 10;
                if (++pointer == 3 && n > 0)
                {
                    buffer.Append('.');
                    pointer = 0;
                }
            }

            return new string(Reverse(buffer).ToArray());
        }

        IEnumerable<char> Reverse(StringBuilder buffer)
        {
            for (var i = buffer.Length - 1; i >= 0; i--)
                yield return buffer[i];
        }

        public static int PossibleStringCount(string word)
        {
            var result = 1;
            var multiplier = 1;
            for (var i = 0; i < word.Length - 1; i++)
            {
                if (word[i] == word[i + 1])
                {
                    multiplier++;
                }
                else
                {
                    result *= multiplier;
                    multiplier = 1;
                }
            }

            result *= multiplier;
            return result + 1;
        }

        public static int FindSpecialInteger(int[] arr)
        {
            var dict = new Dictionary<int, double>();
            foreach (var x in arr)
            {
                if (!dict.ContainsKey(x))
                    dict.Add(x, 0);
                dict[x] += 100 * (1.0 / arr.Length);
                if (dict[x] >= 25)
                    return x;
            }

            return dict.First(x => x.Value >= 25).Key;
        }

        public static string LongestNiceSubstring(string s)
        {
            var line = s.ToLower();
            var previous = line[0];
            var maxLength = 0;
            var length = 1;
            var maxStart = 0;
            var start = 0;
            for (var i = 1; i < line.Length; i++)
            {
                if (line[i] == previous)
                {
                    length++;
                    if (length > maxLength)
                    {
                        maxLength = length;
                        maxStart = start;
                    }
                }
                else
                {
                    start = i;
                    length = 1;
                }

                previous = line[i];
            }

            return new string(Enumerable.Range(maxStart, maxLength).Select(x => s[x]).ToArray());
        }

        public static int CountSquares(int[][] matrix)
        {
            var maxLength = Math.Min(matrix.Length, matrix[0].Length);
            var result = 0;
            for (var length = 1; length <= maxLength; length++)
            {
                var zeroFound = false;
                for (var y = 0; y < length; y++)
                {
                    if (zeroFound)
                        break;
                    for (var x = 0; x < length; x++)
                    {
                        if (matrix[y][x] == 0)
                        {
                            zeroFound = true;
                            break;
                        }
                    }

                    result++;
                }
            }

            return result;
        }

        public int FindClosestNumber(int[] nums)
        {
            var minDistance = int.MaxValue;
            var maxNumber = int.MinValue;
            foreach (var x in nums)
            {
                if (Math.Abs(x) < minDistance || (Math.Abs(x) == minDistance && x >= 0))
                {
                    minDistance = Math.Abs(x);
                    maxNumber = x;
                }
            }

            return maxNumber;
        }

        public IList<string> RemoveSubfolders(string[] folder)
        {
            Array.Sort(folder);
            var deleted = new HashSet<int>(folder.Length);
            for (var i = 0; i < folder.Length; i++)
            {
                if (deleted.Contains(i))
                    continue;
                for (var j = i + 1; j < folder.Length; j++)
                {
                    if (deleted.Contains(j))
                        continue;
                    if (Contains(folder[i], folder[j]))
                        deleted.Add(j);
                }
            }

            var result = new List<string>(folder.Length - deleted.Count);
            for (var i = 0; i < folder.Length; i++)
                if (!deleted.Contains(i))
                    result.Add(folder[i]);
            return result;
        }

        bool Contains(string x, string y)
        {
            if (y.Length < x.Length)
                return false;
            for (var i = 0; i < x.Length; i++)
                if (y[i] != x[i])
                    return false;
            if (y.Length > x.Length)
                return y[x.Length] == '/';
            return true;
        }

        public static int ArrangeCoins(int n)
        {
            var current = 0;
            var sum = 0;
            for (var i = 1; i <= n; i++)
            {
                current += 1;
                sum += current;
                if (sum > n)
                    return i - 1;
                if (sum == n)
                    return i;
            }

            return n;
        }

        public int CountTime(string time)
        {
            var m1 = 1;
            if (time[0] == '?' && time[1] == '?')
                m1 = 24;
            else if (time[0] == '?')
                m1 = time[1] < '4' ? 3 : 2;
            else if (time[1] == '?')
                m1 = time[0] < '2' ? 10 : 4;
            var m2 = 1;
            if (time[3] == '?')
                m2 = 6;
            if (time[4] == '?')
                m2 *= 10;
            return m1 * m2;
        }

        public bool IsGood(int[] nums)
        {
            var max = nums.Max();
            if (nums.Length > max + 1)
                return false;
            var counts = new int[max];
            foreach (var x in nums)
            {
                if (x < 1 || x > max)
                    return false;
                counts[x - 1]++;
                if (counts[x - 1] > 1 && x != max)
                    return false;
            }

            for (var i = 0; i < counts.Length - 1; i++)
            {
                if (counts[i] != 1)
                    return false;
            }

            return counts[^1] == 2;
        }

        public int CountOdds(int low, int high)
        {
            return high % 2 + low % 2 + (high - high % 2 - low - low % 2 - 1) / 2;
        }

        public static int FindLengthOfLCIS(int[] nums)
        {
            var maxLength = 1;
            var previous = nums[0];
            var length = 1;
            for (var i = 1; i < nums.Length; i++)
            {
                if (nums[i] > previous)
                {
                    length++;
                    maxLength = Math.Max(maxLength, length);
                }
                else
                {
                    length = 1;
                }

                previous = nums[i];
            }

            return maxLength;
        }

        public int CountBeautifulPairs(int[] nums)
        {
            var count = 0;
            for (var i = 0; i < nums.Length; i++)
            {
                var x = nums[i].ToString().First() - '0';
                for (var j = i + 1; j < nums.Length; j++)
                {
                    var y = nums[j] % 10;
                    if (Gcd(x, y) == 1)
                        count++;
                }
            }

            return count;
        }

        static int Gcd(int x, int y)
        {
            while (true)
            {
                if (y == 0) return x;
                var x1 = x;
                x = y;
                y = x1 % y;
            }
        }

        public int DaysBetweenDates(string date1, string date2)
        {
            return Math.Abs((DateTime.Parse(date1) - DateTime.Parse(date2)).Days);
        }

        public int MostFrequentEven(int[] nums)
        {
            var numbers = new Dictionary<int, int>();
            var maxFrequency = -1;
            foreach (var x in nums)
            {
                if (x % 2 != 0)
                    continue;
                if (!numbers.ContainsKey(x))
                    numbers.Add(x, 0);
                numbers[x]++;
                maxFrequency = Math.Max(maxFrequency, numbers[x]);
            }

            if (maxFrequency == -1)
                return -1;

            return numbers.Where(x => x.Value == maxFrequency).Min(x => x.Key);
        }

        public static int SecondHighest(string s)
        {
            var max = 0;
            foreach (var x in s)
            {
                if (!char.IsDigit(x))
                    continue;
                max = Math.Max(max, x - '0');
            }

            if (max == 0)
                return -1;
            var nextMax = -1;
            var minDifference = int.MaxValue;
            for (var i = 0; i < s.Length; i++)
            {
                if (!char.IsDigit(s[i]))
                    continue;
                if (s[i] - '0' < max && max - s[i] + '0' < minDifference)
                {
                    minDifference = max - s[i] + '0';
                    nextMax = s[i] - '0';
                }
            }

            return nextMax;
        }

        public static int MaxOperations(int[] nums)
        {
            var score = nums[0] + nums[1];
            var operations = 1;
            for (var i = 2; i < nums.Length - 1; i += 2)
            {
                if (nums[i] + nums[i + 1] != score)
                    return operations;
                operations++;
            }

            return operations;
        }

        public char NextGreatestLetter(char[] letters, char target)
        {
            var min = int.MaxValue;
            var value = '\0';
            foreach (var x in letters)
            {
                if (x - target <= 0)
                    continue;
                if (x - target < min)
                {
                    min = x - target;
                    value = x;
                }
            }

            return value;
        }

        public static int[] GetNoZeroIntegers(int n)
        {
            var x = 0;
            var multiplier = 1;
            for (var i = 0; i < n.ToString().Length; i++)
            {
                if (x > 0 && n / multiplier % 10 == 1 && i == n.ToString().Length - 1)
                    break;
                x += multiplier;
                n -= multiplier;
                var digit = n / multiplier % 10;
                if (digit == 0)
                {
                    x += multiplier;
                    n -= multiplier;
                }

                multiplier *= 10;
            }

            return new[] { n, x };
        }

        public static int MaximumSwap(int num)
        {
            var number = num.ToString().Select(x => x - '0').ToArray();
            for (var i = 0; i < number.Length; i++)
            {
                var max = int.MinValue;
                var maxIndex = i;
                for (var j = i + 1; j < number.Length; j++)
                {
                    if (number[j] > number[i] && number[j] >= max)
                    {
                        max = number[j];
                        maxIndex = j;
                    }
                }

                if (maxIndex != i)
                {
                    var x = number[maxIndex] * (int)Math.Pow(10, number.Length - maxIndex - 1);
                    var y = number[i] * (int)Math.Pow(10, number.Length - i - 1);
                    var u = number[maxIndex] * (int)Math.Pow(10, number.Length - i - 1);
                    var w = number[i] * (int)Math.Pow(10, number.Length - maxIndex - 1);
                    return num - x - y + u + w;
                }
            }

            return num;
        }

        public bool IsFascinating(int n)
        {
            var x = n;
            var y = 2 * n;
            var z = 3 * n;
            var buffer = new int[10];
            while (x > 0)
            {
                buffer[x % 10]++;
                x /= 10;
            }

            while (y > 0)
            {
                buffer[y % 10]++;
                y /= 10;
            }

            while (z > 0)
            {
                buffer[z % 10]++;
                z /= 10;
            }

            return buffer[0] == 0 && buffer.Skip(1).All(x => x == 1);
        }

        public static string AddStrings(string num1, string num2)
        {
            if (num1 == "0" && num2 == "0")
                return "0";
            if (num2.Length > num1.Length)
                (num1, num2) = (num2, num1);
            var sb = new StringBuilder(num1.Length * 10);
            var i = 0;
            var remainder = 0;
            while (true)
            {
                var x = num1.Length - 1 - i >= 0 ? num1[num1.Length - 1 - i] - '0' : 0;
                var y = num2.Length - 1 - i >= 0 ? num2[num2.Length - 1 - i] - '0' : 0;
                var current = x + y + remainder;
                if (current == 0 && i >= num1.Length && remainder == 0)
                    break;
                sb.Append(current % 10);
                remainder = current / 10;
                i++;
            }

            var result = new StringBuilder(sb.Length);
            for (var j = sb.Length - 1; j >= 0; j--)
            {
                result.Append(sb[j]);
            }

            return result.ToString();
        }

        public static bool HaveConflict(string[] event1, string[] event2)
        {
            if (event1[0].CompareTo(event2[0]) == -1)
            {
                return event1[1].CompareTo(event2[0]) > -1;
            }

            if (event1[0].CompareTo(event2[0]) == 1)
                return event2[1].CompareTo(event1[0]) > -1;
            return true;
        }

        public string LosingPlayer(int x, int y)
        {
            return Math.Min(y / 4, x) % 2 == 0 ? "Alice" : "Bob";
        }

        public static bool Check(int[] nums)
        {
            var minIndex = -1;
            var min = int.MaxValue;
            for (var i = 0; i < nums.Length; i++)
            {
                if (nums[i] < min)
                {
                    min = nums[i];
                    minIndex = i;
                }
            }

            var j = minIndex;
            var count = 0;
            var prev = int.MinValue;
            while (count < nums.Length - 1)
            {
                if (count > 0 && nums[j] == prev)
                {
                    count++;
                    j++;
                    continue;
                }

                var current = j % nums.Length;
                var next = (j + 1) % nums.Length;
                if (nums[next] < nums[current])
                    return false;
                count++;
                j++;
                prev = nums[j];
            }

            return true;
        }

        public static string Reformat(string s)
        {
            var letters = new List<char>(s.Length);
            var digits = new List<char>(s.Length);
            foreach (var x in s)
            {
                if (char.IsDigit(x))
                    digits.Add(x);
                else if (char.IsLetter(x))
                    letters.Add(x);
            }

            if (Math.Abs(letters.Count - digits.Count) > 1)
                return string.Empty;
            var primary = digits;
            var secondary = letters;
            if (letters.Count > digits.Count)
            {
                primary = letters;
                secondary = digits;
            }

            var sb = new StringBuilder(s.Length);
            for (var i = 0; i < s.Length; i++)
            {
                if (i % 2 == 0)
                    sb.Append(primary[i / 2]);
                else
                    sb.Append(secondary[i / 2]);
            }

            return sb.ToString();
        }

        public bool StrongPasswordCheckerII(string password)
        {
            if (password.Length < 8)
                return false;
            var hasLowercase = false;
            var hasUppercase = false;
            var hasDigit = false;
            var hasSpecialCharacter = false;
            var special = new HashSet<char>("!@#$%^&*()-+");
            var prev = '\0';
            foreach (var x in password)
            {
                if (char.IsDigit(x))
                    hasDigit = true;
                else if (char.IsLower(x))
                    hasLowercase = true;
                else if (char.IsUpper(x))
                    hasUppercase = true;
                else if (special.Contains(x))
                    hasSpecialCharacter = true;
                if (prev == x)
                    return false;
                prev = x;
            }

            return hasLowercase && hasUppercase && hasDigit && hasSpecialCharacter;
        }

        public static int GetMinDistance(int[] nums, int target, int start)
        {
            if (target == nums[start])
                return 0;
            var min = int.MaxValue;
            for (var i = 0; i < nums.Length; i++)
            {
                if (nums[i] != target)
                    continue;
                var x = start;
                var y = i;
                if (y < x)
                    (y, x) = (x, y);
                min = Math.Min(y - x, min);
            }

            return min;
        }

        public static int MaxWidthRamp(int[] nums)
        {
            var max = int.MinValue;
            var sorted = nums
                .Select((x, i) => new { Number = x, Index = i })
                .OrderBy(x => x.Number)
                .Select(x => x.Index)
                .ToArray();
            var j = 0;
            for (var i = sorted.Length - 1; i >= 0; i--)
            {
                j = Math.Max(j, sorted[i]);
                max = Math.Max(j - sorted[i], max);
            }

            return max == int.MinValue ? 0 : max;
        }

        public int MinimumDifference(int[] nums, int k)
        {
            Array.Sort(nums);
            var min = int.MaxValue;
            if (k == 1)
                return nums[0];
            for (var i = 0; i < nums.Length - 1; i++)
            {
                min = Math.Min(min, nums[i + 1] - nums[i]);
            }

            return min;
        }

        public static IList<string> RemoveAnagrams(string[] words)
        {
            var result = new List<string>(words.Length);
            var previous = string.Empty;
            foreach (var x in words)
            {
                var ordered = Get(x);
                if (ordered == previous)
                    continue;
                previous = ordered;
                result.Add(x);
            }

            return result;
        }

        static string Get(string word)
        {
            var sb = new StringBuilder("00000000000000000000000000");
            foreach (var x in word)
            {
                sb[x - 'a']++;
            }

            return sb.ToString();
        }

        public static bool KLengthApart(int[] nums, int k)
        {
            var lastIndex = 0;
            var firstOneFound = false;
            for (var i = 0; i < nums.Length; i++)
            {
                if (nums[i] != 1)
                    continue;
                if (!firstOneFound)
                    firstOneFound = true;
                else if (i - lastIndex - 1 < k)
                    return false;

                lastIndex = i;
            }

            return true;
        }

        public int FillCups(int[] amount)
        {
            var count = 0;
            while (true)
            {
                Array.Sort(amount);
                if (amount[1] == 0)
                    return count + amount[2];
                amount[1]--;
                amount[2]--;
                count++;
            }
        }

        public IList<string> LetterCombinations(string digits)
        {
            if (digits.Length == 0)
                return new List<string>();
            var dict = new Dictionary<char, char[]>()
            {
                { '2', new char[] { 'a', 'b', 'c' } },
                { '3', new char[] { 'd', 'e', 'f' } },
                { '4', new char[] { 'g', 'h', 'i' } },
                { '5', new char[] { 'j', 'k', 'l' } },
                { '6', new char[] { 'm', 'n', 'o' } },
                { '7', new char[] { 'p', 'q', 'r', 's' } },
                { '8', new char[] { 't', 'u', 'v' } },
                { '9', new char[] { 'w', 'x', 'y', 'z' } },
            };
            var result = new List<string>();
            foreach (var x in dict[digits[0]])
            {
                if (digits.Length == 1)
                {
                    result.Add(new string(new[] { x }));
                    continue;
                }

                foreach (var y in dict[digits[1]])
                {
                    if (digits.Length == 2)
                    {
                        result.Add(new string(new[] { x, y }));
                        continue;
                    }

                    foreach (var u in dict[digits[2]])
                    {
                        if (digits.Length == 3)
                        {
                            result.Add(new string(new[] { x, y, u }));
                            continue;
                        }

                        foreach (var w in dict[digits[3]])
                            result.Add(new string(new[] { x, y, u, w }));
                    }
                }
            }

            return result;
        }

        public IEnumerable<IEnumerable<T>> CartesianProduct<T>(IEnumerable<IEnumerable<T>> sequences)
        {
            IEnumerable<IEnumerable<T>> emptyProduct = new[] { Enumerable.Empty<T>() };
            return sequences.Aggregate(
                emptyProduct,
                (accumulator, sequence) =>
                    accumulator.SelectMany(accSeq =>
                        sequence.Select(item => accSeq.Concat(new[] { item }))));
        }

        public char SlowestKey(int[] releaseTimes, string keysPressed)
        {
            var dict = new Dictionary<char, int>();
            for (var i = 0; i < keysPressed.Length; i++)
            {
                var dt = i == 0
                    ? releaseTimes[0]
                    : releaseTimes[i] - releaseTimes[i - 1];
                if (!dict.ContainsKey(keysPressed[i]))
                    dict.Add(keysPressed[i], 0);
                dict[keysPressed[i]] = Math.Max(dict[keysPressed[i]], dt);
            }

            return dict.OrderByDescending(x => x.Value)
                .ThenByDescending(x => x.Key)
                .First().Key;
        }

        public static bool RotateString(string source, string goal)
        {
            if (source.Length != goal.Length)
                return false;
            for (var shift = 0; shift < source.Length; shift++)
            {
                var matched = true;
                for (var i = 0; i < source.Length; i++)
                {
                    if (source[(i + shift) % source.Length] != goal[i])
                    {
                        matched = false;
                        break;
                    }
                }

                if (matched)
                    return true;
            }

            return false;
        }

        public static int FindMaxConsecutiveOnes(int[] nums)
        {
            var maxLength = 0;
            var startIndex = 0;
            var found = false;
            for (var i = 0; i < nums.Length; i++)
            {
                if (i == 0 && nums[0] == 1)
                {
                    startIndex = 0;
                    maxLength = Math.Max(maxLength, i - startIndex + 1);
                    found = true;
                    continue;
                }

                if (i == 0 && nums[0] == 0)
                {
                    continue;
                }

                if (nums[i - 1] == 0 && nums[i] == 1)
                {
                    startIndex = i;
                    found = true;
                    maxLength = Math.Max(maxLength, i - startIndex + 1);
                    continue;
                }

                if (nums[i] == 0)
                {
                    if (found)
                        maxLength = Math.Max(maxLength, i - startIndex);
                    found = false;
                }

                if (i == nums.Length - 1 && nums[i] == 1)
                {
                    maxLength = Math.Max(maxLength, i - startIndex + 1);
                }
            }

            return maxLength;
        }

        public static string CapitalizeTitle(string title)
        {
            var sb = new StringBuilder(title.Length);
            for (var i = 0; i < title.Length; i++)
            {
                if (i != 0)
                    sb.Append(' ');
                var pointer = i;
                while (pointer < title.Length && title[pointer] != ' ')
                {
                    sb.Append(title[pointer] <= 'Z' ? (char)(title[pointer] - 'A' + 'a') : title[pointer]);
                    pointer++;
                }

                if (pointer - i > 2)
                {
                    sb[sb.Length - pointer + i] = (char)(sb[sb.Length - pointer + i] - 'a' + 'A');
                }

                i += pointer - i;
            }

            return sb.ToString();
        }

        public int[] ArrayRankTransform(int[] arr)
        {
            var dict = arr.Distinct().OrderBy(x => x).Select((x, i) => new { Number = x, Order = i + 1 })
                .ToDictionary(x => x.Number, x => x.Order);
            for (var i = 0; i < arr.Length; i++)
                arr[i] = dict[arr[i]];
            return arr;
        }

        public bool IsPossibleToSplit(int[] nums)
        {
            var dict = new Dictionary<int, int>();
            foreach (var x in nums)
            {
                if (!dict.ContainsKey(x))
                    dict.Add(x, 0);
                dict[x]++;
                if (dict[x] > 2)
                    return false;
            }

            var solitaryNumbers = 0;
            foreach (var x in dict.Values)
            {
                if (x == 1)
                    solitaryNumbers++;
            }

            return solitaryNumbers % 2 == 0;
        }

        public int MinNumber(int[] nums1, int[] nums2)
        {
            var intersection = nums1.ToHashSet().Intersect(nums2.ToHashSet()).ToArray();
            if (intersection.Any())
                return intersection.Min();
            var minDigit1 = nums1.Min();
            var minDigit2 = nums2.Min();
            if (minDigit1 < minDigit2)
                return 10 * minDigit1 + minDigit2;
            return 10 * minDigit2 + minDigit1;
        }

        public static int FindShortestSubArray(int[] nums)
        {
            var visited = new HashSet<int>();
            var frequency = new Dictionary<int, int>();
            var distance = new Dictionary<int, int>();
            for (var i = 0; i < nums.Length; i++)
            {
                if (!frequency.ContainsKey(nums[i]))
                    frequency.Add(nums[i], 0);
                frequency[nums[i]]++;
                if (!distance.ContainsKey(nums[i]))
                    distance.Add(nums[i], 1);
                if (visited.Contains(nums[i]))
                    continue;
                for (var j = i + 1; j < nums.Length; j++)
                {
                    if (nums[i] != nums[j])
                        continue;
                    distance[nums[i]] = j - i + 1;
                }

                visited.Add(nums[i]);
            }

            var minDistance = int.MaxValue;
            var maxFrequency = frequency.Max(x => x.Value);
            foreach (var f in frequency.Where(x => x.Value == maxFrequency))
            {
                minDistance = Math.Min(distance[f.Key], minDistance);
            }

            return minDistance;
        }

        public int MaximumDifference(int[] nums)
        {
            var diff = -1;
            for (var i = 0; i < nums.Length; i++)
            for (var j = i + 1; j < nums.Length; j++)
            {
                if (nums[i] < nums[j])
                    diff = Math.Max(nums[j] - nums[i], diff);
            }

            return diff;
        }

        public string DayOfTheWeek(int day, int month, int year)
        {
            var dateTime = new DateTime(year, month, day);
            return dateTime.DayOfWeek.ToString();
        }

        public static int FindKthPositive(int[] arr, int k)
        {
            var missingNumbers = arr[0] - 1;
            if (missingNumbers >= k)
                return arr[0] - 1 - missingNumbers;

            for (var i = 0; i < arr.Length - 1; i++)
            {
                var dx = arr[i + 1] - arr[i] - 1;
                if (missingNumbers + dx >= k)
                {
                    return arr[i] + (k - missingNumbers);
                }

                missingNumbers += dx;
            }

            return arr[^1] + k;
        }

        public bool IsMonotonic(int[] nums)
        {
            if (nums.Length == 1)
                return true;
            var decreasing = false;
            for (var i = 0; i < nums.Length - 1; i++)
            {
                var delta = nums[i + 1] - nums[i];
                if (delta < 0)
                {
                    decreasing = true;
                }

                if (decreasing)
                {
                    if (delta > 0)
                        return false;
                }
                else
                {
                    if (delta < 0)
                        return false;
                }
            }

            return true;
        }

        public static int WinningPlayerCount(int n, int[][] pick)
        {
            var balls = new Dictionary<int, int[]>();
            var winners = 0;
            foreach (var x in pick)
            {
                if (!balls.ContainsKey(x[0]))
                    balls.Add(x[0], new int[11]);
                balls[x[0]][x[1]]++;
            }

            for (var player = 0; player < n; player++)
            {
                if (!balls.ContainsKey(player))
                    continue;
                var isPlayerDefeat = true;
                for (var i = 0; i < 11; i++)
                {
                    if (balls[player][i] == 0)
                        continue;
                    if (balls[player][i] > player)
                    {
                        isPlayerDefeat = false;
                        break;
                    }
                }


                if (!isPlayerDefeat)
                    winners++;
            }

            return winners;
        }

        public int CountQuadruplets(int[] nums)
        {
            var n = nums.Length;
            var quadruples = 0;
            for (var d = n - 1; d > 2; d--)
            {
                for (var c = d - 1; c > 1; c--)
                for (var b = c - 1; b > 0; b--)
                for (var a = b - 1; a >= 0; a--)
                {
                    if (nums[a] + nums[b] + nums[c] != nums[d])
                        continue;
                    quadruples++;
                }
            }

            return quadruples;
        }

        public static int MinimumRightShifts(IList<int> nums)
        {
            var shiftingIndex = -1;
            for (var i = 1; i < nums.Count; i++)
            {
                if (nums[i] < nums[i - 1])
                {
                    shiftingIndex = i;
                    break;
                }
            }

            if (shiftingIndex == -1)
                return 0;
            for (var i = shiftingIndex; i < nums.Count; i++)
            {
                if (nums[i] > nums[0] || (nums[i] < nums[i - 1] && i > shiftingIndex))
                    return -1;
            }

            return (nums.Count - shiftingIndex) % nums.Count;
        }

        public int MaxSum(int[] nums)
        {
            var maxSum = -1;
            var buffer = new Dictionary<int, int>();
            foreach (var x in nums)
            {
                var maxDigit = MaxDigit(x);
                if (buffer.ContainsKey(maxDigit))
                {
                    maxSum = Math.Max(buffer[maxDigit] + x, maxSum);
                    buffer[maxDigit] = Math.Max(buffer[maxDigit], x);
                }
                else
                    buffer.Add(maxDigit, x);
            }

            return maxSum;
        }

        int MaxDigit(int number)
        {
            var max = 0;
            while (number > 0)
            {
                max = Math.Max(max, number % 10);
                number /= 10;
            }

            return max;
        }

        public int FindKthNumber(int n, int k)
        {
            return Enumerable.Range(0, n)
                .Select(x => new { Number = x, String = x.ToString() })
                .OrderBy(x => x.String)
                .Skip(k - 1)
                .First().Number;
        }

        public static IList<int> LexicalOrder(int n)
        {
            return Enumerable.Range(1, n)
                .Select(x => new { Number = x, String = x.ToString() })
                .OrderBy(x => x.String)
                .Select(x => x.Number)
                .ToArray();
        }

        public string ShortestPalindrome(string word)
        {
            var buffer = new StringBuilder(word);
            var prefix = new StringBuilder(word.Length);
            for (var i = buffer.Length - 1; i > 0; i--)
            {
                if (IsPalindrome(buffer.ToString()))
                    break;
                prefix.Append(buffer[i]);
                buffer.Remove(buffer.Length - 1, 1);
            }

            return prefix + word;
        }

        bool IsPalindrome(string word)
        {
            if (word.Length == 1)
                return false;
            for (var i = 0; i < word.Length / 2; i++)
                if (word[i] != word[word.Length - 1 - i])
                    return false;
            return true;
        }

        public int MinimumSum(int[] nums)
        {
            var max = int.MaxValue;
            for (var i = 0; i < nums.Length; i++)
            for (var j = i + 1; j < nums.Length; j++)
            for (var k = j + 1; k < nums.Length; k++)
            {
                if (nums[i] < nums[j] && nums[k] < nums[j])
                    max = Math.Min(max, nums[i] + nums[j] + nums[k]);
            }

            return max == int.MaxValue ? -1 : max;
        }

        // public string LargestNumber(int[] nums)
        // {
        //     var a = nums.Select(x => x.ToString())
        //         .
        // }

        public int MinimumCost(int[] cost)
        {
            Array.Sort(cost);
            var pointer = 0;
            var sum = 0;
            for (var i = 0; i < cost.Length; i++)
            {
                if (pointer == 2)
                {
                    pointer = 0;
                    continue;
                }

                sum += cost[cost.Length - 1 - i];
                pointer++;
            }

            return sum;
        }

        public static int KItemsWithMaximumSum(int numOnes, int numZeros, int numNegOnes, int k)
        {
            var sum = 0;
            var count = 0;
            for (var i = 0; i < numOnes && i < k; i++)
            {
                sum++;
                count++;
            }

            for (var i = 0; i < numZeros && count < k; i++)
                count++;
            for (var i = 0; i < numNegOnes && count < k; i++)
                sum--;
            return sum;
        }

        public bool CanConstruct(string ransomNote, string magazine)
        {
            var letters = new int[26];
            foreach (var x in magazine)
                letters[x - 'a']++;
            foreach (var x in ransomNote)
            {
                if (letters[x - 'a'] == 0)
                    return false;
                letters[x - 'a']--;
            }

            return true;
        }

        public IList<string> StringMatching(string[] words)
        {
            words = words.OrderBy(x => x.Length).ToArray();
            var result = new List<string>(words.Length);
            for (var i = 0; i < words.Length; i++)
            for (var j = 0; j < words.Length; j++)
                if (words[j].Contains(words[i]))
                    result.Add(words[j]);
            return result;
        }

        public int FindMinDifference(IList<string> timePoints)
        {
            var day = 24 * 60;
            var timeInMinutes = timePoints
                .Select(DateTime.Parse)
                .Select(x => x.Hour * 60 + x.Minute)
                .ToArray();
            var min = day;
            for (var i = 0; i < timeInMinutes.Length; i++)
            {
                for (var j = i + 1; j < timeInMinutes.Length; j++)
                {
                    var difference = Math.Abs(timeInMinutes[j] - timeInMinutes[i]);
                    if (difference > day / 2)
                        difference = day - difference;
                    min = Math.Min(min, difference);
                }
            }

            return min;
        }

        public static int ConvertTime(string time1, string time2)
        {
            var current = DateTime.Parse(time1);
            var correct = DateTime.Parse(time2);
            var minutes = correct.Hour * 60 + correct.Minute - current.Hour * 60 - current.Minute;
            // var denominators = new[] { 60, 15, 5, 1 };
            var ticks = 0;
            while (minutes > 0)
            {
                if (minutes / 60 > 0)
                {
                    ticks += minutes / 60;
                    minutes %= 60;
                }
                else if (minutes / 15 > 0)
                {
                    ticks += minutes / 15;
                    minutes %= 15;
                }
                else if (minutes / 5 > 0)
                {
                    ticks += minutes / 5;
                    minutes %= 5;
                }
                else if (minutes / 1 > 0)
                {
                    ticks += minutes / 1;
                    minutes %= 1;
                }
            }

            return ticks;
        }

        public static bool CheckAlmostEquivalent(string word1, string word2)
        {
            var freq = new int[26];
            foreach (var x in word1)
                freq[x - 'a']++;
            foreach (var x in word2)
            {
                if (Math.Abs(freq[x - 'a']) > 3)
                    return false;
                freq[x - 'a']--;
                if (Math.Abs(freq[x - 'a']) > 3)
                    return false;
            }

            return true;
        }

        public int MinimumBoxes(int[] apple, int[] capacity)
        {
            Array.Sort(capacity);
            var boxes = 0;
            var apples = apple.Sum();
            var sumCapacity = 0;
            for (var i = 0; i < capacity.Length; i++)
            {
                sumCapacity += capacity[capacity.Length - 1 - i];
                boxes++;
                if (sumCapacity >= apples)
                    break;
            }

            return boxes;
        }

        public int MaxLengthBetweenEqualCharacters(string s)
        {
            var max = 0;
            for (var i = 0; i < s.Length; i++)
            for (var j = i + 1; j < s.Length; j++)
                if (s[i] == s[j])
                {
                    var length = j - i - 1;
                    max = Math.Max(length, max);
                }

            return max;
        }

        public static int AlternateDigitSum(int n)
        {
            var alternateSum = 0;
            var x = n;
            var lastIndex = 0;
            for (var i = 0; i < 31 && x > 0; i++)
            {
                lastIndex = i;
                x /= 10;
            }

            var multiplier = 1;
            if (lastIndex % 2 == 1)
                multiplier = -1;
            for (var i = 0; i < 31; i++)
            {
                var digit = n % 10;
                alternateSum += digit * multiplier;
                n /= 10;
                multiplier *= -1;
            }

            return alternateSum;
        }

        public static int CountCharacters(string[] words, string chars)
        {
            var dict = new int[26];
            foreach (var x in chars)
                dict[x - 'a']++;

            var result = 0;
            foreach (var word in words)
            {
                var buffer = new int[26];
                foreach (var x in word)
                    buffer[x - 'a']++;
                var isSubset = true;
                for (var i = 0; i < 26; i++)
                    if (dict[i] < buffer[i])
                    {
                        isSubset = false;
                        break;
                    }

                if (isSubset)
                    result += word.Length;
            }

            return result;
        }

        public bool IsPrefixString(string line, string[] words)
        {
            var wordsPointer = 0;
            var letterPointer = 0;
            for (var i = 0; i < line.Length; i++)
            {
                if (line[i] != words[wordsPointer][letterPointer])
                    return false;
                if (i == line.Length - 1)
                {
                    if (letterPointer + 1 != words[wordsPointer].Length)
                        return false;
                    return true;
                }

                letterPointer++;
                if (letterPointer == words[wordsPointer].Length)
                {
                    letterPointer = 0;
                    wordsPointer++;
                }

                if (wordsPointer == words.Length)
                    return false;
            }

            return true;
        }

        public static int DistinctAverages(int[] nums)
        {
            Array.Sort(nums);
            var averages = new HashSet<double>(nums.Length / 2);
            for (var i = 0; i < nums.Length / 2 + 1; i++)
            {
                averages.Add(((double)nums[i] + nums[nums.Length - 1 - i]) / 2);
            }

            return averages.Count;
        }

        public bool IsAcronym(IList<string> words, string s)
        {
            if (s.Length != words.Count)
                return false;
            for (var i = 0; i < s.Length; i++)
            {
                if (words[i][0] != s[i])
                    return false;
            }

            return true;
        }

        public IList<IList<int>> MergeSimilarItems(int[][] items1, int[][] items2)
        {
            var dict = new Dictionary<int, int>();
            foreach (var x in items1)
            {
                if (!dict.ContainsKey(x[0]))
                    dict.Add(x[0], 0);
                dict[x[0]] += x[1];
            }

            foreach (var x in items2)
            {
                if (!dict.ContainsKey(x[0]))
                    dict.Add(x[0], 0);
                dict[x[0]] += x[1];
            }

            return dict.Select(x => new List<int> { x.Key, x.Value })
                .OrderBy(x => x[0])
                .ToArray();
        }

        static void GetKangarooWords()
        {
            // var n = int.Parse(Console.ReadLine());
            var n = 1;
            // var line = Console.ReadLine();
            var line = "alone, lone, one";
            for (int i = 0; i < n; i++)
            {
                var result = ParseKangarooWords(line.Split(", "));
                if (result.Count == 0)
                    Console.WriteLine("NONE");
                Console.WriteLine(string.Join("\n", result));
            }
        }

        static List<string> ParseKangarooWords(string[] input)
        {
            var words = input
                .OrderBy(x => x.Length)
                .ThenBy(x => x)
                .ToArray();
            var added = new HashSet<int>();
            var list = new List<List<string>>();
            for (var i = 0; i < words.Length; i++)
            {
                if (added.Contains(i))
                    continue;
                list.Add(new List<string> { words[i] });
                for (var j = i + 1; j < words.Length; j++)
                {
                    if (added.Contains(j))
                        continue;
                    if (words[j].Contains(words[i]))
                    {
                        list[i].Add(words[j]);
                        added.Add(j);
                    }
                }
            }

            var result = new List<string>();
            foreach (var x in list)
            {
                if (x.Count < 2)
                    continue;
                result.Add($"{x.Last()}: {string.Join(", ", x.Take(x.Count - 1).OrderBy(y => y.Length))}");
            }

            return result;
        }

        public int CanBeTypedWords(string text, string brokenLetters)
        {
            var broken = new HashSet<char>(brokenLetters);
            var words = text.Split(' ');
            var canBeTypedCount = words.Length;
            foreach (var word in words)
            {
                foreach (var x in word)
                {
                    if (broken.Contains(x))
                    {
                        canBeTypedCount--;
                        break;
                    }
                }
            }

            return canBeTypedCount;
        }

        public static int GetLucky(string line, int k)
        {
            long number = 0;
            var sb = new StringBuilder(line.Length * 3);
            foreach (var x in line)
            {
                sb.Append(x - 'a' + 1);
            }

            for (var i = 0; i < k; i++)
            {
                var x = 0;
                for (var j = 0; j < sb.Length; j++)
                    x += sb[j] - '0';

                sb = new StringBuilder();
                sb.Append(x);
            }

            return int.Parse(sb.ToString());
        }

        public int[] DistributeCandies(int candies, int peopleCount)
        {
            var result = new int[peopleCount];
            var pointer = 0;
            var current = 0;
            while (candies > 0)
            {
                result[pointer] = Math.Min(1 + pointer, candies);
                candies -= current;
                pointer++;
                pointer %= peopleCount;
            }

            return result;
        }

        public int NumOfStrings(string[] patterns, string word)
        {
            var checkedPatterns = new HashSet<string>();
            var count = 0;
            foreach (var x in patterns)
            {
                if (checkedPatterns.Contains(x))
                    continue;
                if (word.Contains(x))
                    count++;
                else
                    checkedPatterns.Add(x);
            }

            return count;
        }

        public IList<int> FindDisappearedNumbers(int[] nums)
        {
            var array = new int[nums.Length];
            foreach (var x in nums)
                array[x - 1]++;
            return Get(array).ToArray();
        }

        IEnumerable<int> Get(int[] nums)
        {
            for (var i = 0; i < nums.Length; i++)
                if (nums[i] == 0)
                    yield return i + 1;
        }

        public bool HasAlternatingBits(int n)
        {
            var currentValue = -1;
            for (var i = 0; i < 31 && 1 << i <= n; i++)
            {
                var actualValue = (n & (1 << i)) >> i;
                if (actualValue == currentValue)
                    return false;
                currentValue = actualValue;
            }

            return true;
        }

        public static string FractionAddition(string expression)
        {
            var fractions = new List<(int Numerator, int Denominator)>();
            (int Numerator, int Denominator) currentFraction = (0, 0);
            var isNumeratorStage = false;
            var multiplier = 1;
            for (var i = expression.Length - 1; i >= 0; i--)
            {
                if (expression[i] == '+')
                {
                    multiplier = 1;
                    isNumeratorStage = false;
                    fractions.Add(currentFraction);
                    currentFraction = (0, 0);
                }
                else if (expression[i] == '-')
                {
                    multiplier = 1;
                    isNumeratorStage = false;
                    currentFraction.Numerator *= -1;
                    fractions.Add(currentFraction);
                    currentFraction = (0, 0);
                }
                else if (expression[i] == '/')
                {
                    isNumeratorStage = true;
                    multiplier = 1;
                }
                else if (isNumeratorStage)
                {
                    currentFraction.Numerator += multiplier * (expression[i] - '0');
                    multiplier *= 10;
                    if (i == 0)
                        fractions.Add(currentFraction);
                }
                else
                {
                    currentFraction.Denominator += multiplier * (expression[i] - '0');
                    multiplier *= 10;
                }
            }

            if (fractions.Count == 1)
                return expression;

            var commonDenominator = fractions[0].Denominator;
            for (var i = 1; i < fractions.Count; i++)
                commonDenominator = Meow(commonDenominator, fractions[i].Denominator);
            var commonNumerator = 0;
            foreach (var x in fractions)
                commonNumerator += Math.Abs(x.Numerator) * (x.Numerator / Math.Abs(x.Numerator)) *
                                   (commonDenominator / x.Denominator);
            var gcd = Gcd(commonDenominator, Math.Abs(commonNumerator));
            return $"{commonNumerator / gcd}/{commonDenominator / gcd}";
        }

        public static int Meow(int a, int b)
        {
            return a / Gcd(a, b) * b;
        }

        public static int FindComplement(int num)
        {
            var x = 0;
            for (var i = 0; i < 31 && 1 << i <= num; i++)
            {
                var bit = (num & (1 << i)) >> i;
                if (bit == 0)
                {
                    x += 1 << i;
                }
            }

            return x;
        }

        public static int PivotIndex(int[] nums)
        {
            var sum = nums.Sum();
            var leftSum = 0;
            for (var i = 0; i < nums.Length; i++)
            {
                if (leftSum == sum - leftSum - nums[i])
                    return i;
                leftSum += nums[i];
            }

            return -1;
        }

        public static bool IsCircularSentence(string sentence)
        {
            if (sentence[^1] != sentence[0])
                return false;
            for (var i = 2; i < sentence.Length; i++)
                if (sentence[i - 1] == ' ' && sentence[i] != sentence[i - 2])
                    return false;
            return true;
        }

        public static int IsPrefixOfWord(string sentence, string searchWord)
        {
            var words = sentence.Split(' ').ToArray();
            for (var i = 0; i < words.Length; i++)
            {
                if (words[i].StartsWith(searchWord))
                    return i + 1;
            }

            return -1;
        }

        public static string LargestOddNumber(string num)
        {
            for (var i = num.Length - 1; i >= 0; i--)
            {
                if ((num[i] - '0') % 2 == 1)
                    return num.Substring(0, i + 1);
            }

            return string.Empty;
        }

        public int LastStoneWeight(int[] stones)
        {
            var queue = new PriorityQueue<int, int>();
            foreach (var x in stones)
                queue.Enqueue(x, -x);
            while (queue.Count > 1)
            {
                var y = queue.Dequeue();
                var x = queue.Dequeue();
                if (x != y)
                    queue.Enqueue(y - x, x - y);
            }

            return queue.Count != 0 ? queue.Dequeue() : 0;
        }

        public int NumberOfSpecialChars(string word)
        {
            var hash = word.ToHashSet();
            return hash.Count(x => x >= 'a' && x <= 'z' && hash.Contains((char)(x + 'A' - 'a')));
        }

        public static int MinChanges(int n, int k)
        {
            var count = 0;
            var min = n;
            var max = k;
            if (n > k)
                (min, max) = (k, n);
            if (n == k)
                return 0;
            for (var i = 0; i < 31; i++)
            {
                var minBit = (min & (1 << i)) >> i;
                var maxBit = (max & (1 << i)) >> i;
                if (minBit != maxBit)
                {
                    if (maxBit == 0)
                        return -1;
                    count++;
                }
            }

            return count;
        }

        public static string ReverseOnlyLetters(string line)
        {
            var sb = new StringBuilder(line);
            var queue = new Queue<int>();
            for (var i = 0; i < line.Length; i++)
            {
                if (!char.IsLetter(line[i]))
                    continue;
                queue.Enqueue(i);
            }

            for (var i = line.Length - 1; i >= 0; i--)
            {
                if (!char.IsLetter(line[i]))
                    continue;
                sb[queue.Dequeue()] = line[i];
            }

            return sb.ToString();
        }

        public int MaxDistance(IList<IList<int>> arrays)
        {
            var max = 0;
            for (var i = 0; i < arrays.Count - 1; i++)
            {
                max = Math.Max(max, arrays[i + 1][arrays[i + 1].Count - 1]);
            }

            for (var i = 0; i < arrays.Count - 1; i++)
            {
                max = Math.Max(max, Math.Abs(arrays[i][arrays[i].Count - 1] - arrays[i + 1][0]));
            }

            return max;
        }

        public static string DigitSum(string source, int length)
        {
            var sb = new StringBuilder(source);
            while (sb.Length > length)
            {
                var sum = 0;
                var buffer = new StringBuilder(length);
                var pointer = 0;
                for (var i = 0; i < sb.Length; i++)
                {
                    var x = sb[i];
                    if (pointer < length)
                        sum += x - '0';
                    else
                    {
                        buffer.Append(sum);
                        sum = x - '0';
                        pointer = 0;
                    }

                    if (i == sb.Length - 1)
                        buffer.Append(sum);
                    pointer++;
                }

                sb = new StringBuilder(buffer.ToString());
            }

            return sb.ToString();
        }

        public static bool LemonadeChange(int[] bills)
        {
            var five = 0;
            var ten = 0;
            foreach (var x in bills)
            {
                if (x == 5)
                    five++;
                if (x == 10)
                    ten++;
                switch (x)
                {
                    case 5:
                        break;
                    case 10 when five > 0:
                        five--;
                        break;
                    case 20 when ten >= 1 && five > 0:
                        ten--;
                        five--;
                        break;
                    case 20 when five >= 3:
                        five -= 3;
                        break;
                    default:
                        return false;
                }
            }

            return true;
        }

        public static int[] Decrypt(int[] code, int k)
        {
            var decoded = new int[code.Length];
            if (k == 0)
                for (var i = 0; i < code.Length; i++)
                    code[i] = 0;
            var multiplier = 1;
            if (k < 0)
                multiplier = -1;
            for (var i = 0; i < code.Length; i++)
            {
                var sum = 0;
                for (var j = 0; j < multiplier * k; j++)
                {
                    sum += code[(i + multiplier * 1 + code.Length + multiplier * j) % code.Length];
                }

                decoded[i] = sum;
            }

            return decoded;
        }

        public int MinimumPushes(string word)
        {
            var dict = new Dictionary<char, int>();
            foreach (var x in word)
            {
                if (!dict.ContainsKey(x))
                    dict.Add(x, 0);
                dict[x]++;
            }

            var pushes = 0;
            var pointer = 0;
            foreach (var kp in dict.OrderByDescending(x => x.Value))
            {
                pushes += (pointer / 8 + 1) * kp.Value;
                pointer++;
            }

            return pushes;
        }

        public long PickGifts(int[] gifts, int k)
        {
            var longs = new long[gifts.Length];
            for (var i = 0; i < gifts.Length; i++)
                longs[i] = gifts[i];
            for (var i = 0; i < k; i++)
            {
                (long Value, int Position) max = (-1, -1);
                for (var t = 0; t < longs.Length; t++)
                {
                    if (longs[t] > max.Value)
                    {
                        max.Value = longs[t];
                        max.Position = t;
                    }
                }

                longs[max.Position] = (long)Math.Floor(Math.Sqrt(max.Value));
            }

            return longs.Sum();
        }

        public int CountEven(int num)
        {
            var count = 0;
            for (var i = 2; i <= num && DigitSum(i) % 2 == 0; i++)
                count++;
            return count;
        }

        int DigitSum(int x)
        {
            var sum = 0;
            while (x > 0)
            {
                sum += x % 10;
                x /= 10;
            }

            return sum;
        }

        public string[] UncommonFromSentences(string s1, string s2)
        {
            var collection1 = s1.Split(" ").GroupBy(x => x).ToDictionary(x => x.Key, x => x.Count());
            var collection2 = s2.Split(" ").GroupBy(x => x).ToDictionary(x => x.Key, x => x.Count());
            var result = new List<string>();
            foreach (var x in collection1)
            {
                if (x.Value > 1)
                {
                    collection1.Remove(x.Key);
                    collection2.Remove(x.Key);
                }
                else if (collection2.ContainsKey(x.Key))
                {
                    collection2.Remove(x.Key);
                }
                else
                {
                    result.Add(x.Key);
                }
            }

            result.AddRange(collection2.Where(x => x.Value == 1).Select(x => x.Key));
            return result.ToArray();
        }

        public int BuyChoco(int[] prices, int money)
        {
            Array.Sort(prices);
            var refund = money - prices[0] - prices[1];
            return refund < 0 ? money : refund;
        }

        public static string? KthDistinct(string[] arr, int k)
        {
            var last = arr
                .Select((w, i) => (w, i))
                .GroupBy(x => x.w)
                .Where(x => x.Count() == 1)
                .SelectMany(x => x)
                .OrderBy(x => x.i)
                .Take(k)
                .ToArray();
            return last.Length == k ? last.Select(x => x.w).LastOrDefault() : "";
        }

        public bool CanBeEqual(int[] target, int[] source)
        {
            var dict1 = target
                .GroupBy(x => x)
                .ToDictionary(x => x.Key, x => x.Count());
            var dict2 = source
                .GroupBy(x => x)
                .ToDictionary(x => x.Key, x => x.Count());
            foreach (var x in dict1.Keys)
            {
                if (!dict2.ContainsKey(x))
                    return false;
                if (dict2[x] != dict1[x])
                    return false;
            }

            return true;
        }

        public string LargestGoodInteger(string line)
        {
            var max = '\0';
            for (var i = 0; i < line.Length - 2; i++)
            {
                if (line[i] == line[i + 1] && line[i + 1] == line[i + 2])
                {
                    if (line[i] > max)
                        max = line[i];
                }
            }

            return max == '\0' ? string.Empty : new string(new[] { max, max, max });
        }

        public bool CanMakeArithmeticProgression(int[] sequence)
        {
            Array.Sort(sequence);
            var dx = 0;
            for (var i = 0; i < sequence.Length - 1; i++)
            {
                if (i == 0)
                {
                    dx = sequence[i + 1] - sequence[i];
                    continue;
                }

                if (dx != sequence[i + 1] - sequence[i])
                    return false;
            }

            return true;
        }

        public int MinLength(string line)
        {
            var sb = new StringBuilder(line);
            while (line.Contains("AB") || line.Contains("CD"))
            {
                line = line.Replace("AB", "");
                line = line.Replace("CD", "");
            }

            return line.Length;
        }

        public string GreatestLetter(string s)
        {
            var hashSet = new HashSet<char>(s);
            var greatest = '\0';
            foreach (var x in hashSet)
            {
                if (x > greatest && hashSet.Contains((char)(x - ('A' - 'a'))))
                    greatest = x;
            }

            return greatest != '\0' ? greatest.ToString() : string.Empty;
        }

        public long FindTheArrayConcVal(int[] nums)
        {
            var left = 0;
            var right = nums.Length - 1;
            var value = 0L;
            while (left <= right)
            {
                if (left == right)
                    value += nums[left];
                else
                {
                    value += long.Parse(nums[left].ToString() + nums[right]);
                }

                left++;
                right--;
            }

            return value;
        }

        public int DistributeCandies(int[] candyType)
        {
            return Math.Min(candyType.ToHashSet().Count, candyType.Length / 2);
        }

        public int SimilarPairs(string[] words)
        {
            var pairs = 0;
            var meow = words.Select(x => x.ToHashSet()).ToArray();
            for (var i = 0; i < meow.Length; i++)
            {
                for (var j = i + 1; j < meow.Length; j++)
                    if (meow[i].SetEquals(meow[j]))
                        pairs++;
            }

            return pairs;
        }

        public int CountWords(string[] words1, string[] words2)
        {
            var count = 0;
            var dict1 = words1
                .GroupBy(x => x)
                .ToDictionary(x => x.Key, x => x.Count());
            var dict2 = words2
                .GroupBy(x => x)
                .ToDictionary(x => x.Key, x => x.Count());
            foreach (var x in dict1.Keys.Where(y => dict2.ContainsKey(y)))
            {
                if (dict1[x] == 1 && dict2[x] == 1)
                    count++;
            }

            return count;
        }

        public static int[] SortJumbled(int[] mapping, int[] nums)
        {
            var dict = new Dictionary<int, int>();
            for (var j = 0; j < nums.Length; j++)
            {
                var x = nums[j];
                var sb = new StringBuilder(x.ToString());
                for (var i = 0; i < sb.Length; i++)
                    sb[i] = (char)(mapping[sb[i] - '0'] + '0');

                dict.TryAdd(x, int.Parse(sb.ToString()));
            }

            return nums.Select(original => (original, dict[original]))
                .OrderBy(x => x.Item2)
                .Select(x => x.original)
                .ToArray();
        }

        public int[] FrequencySort(int[] nums)
        {
            var dict = nums
                .GroupBy(x => x)
                .ToDictionary(x => x.Key, x => x.Count());
            var pointer = 0;
            foreach (var kp in dict.OrderBy(x => x.Value).ThenByDescending(x => x.Key))
            {
                for (var i = 0; i < kp.Value; i++)
                {
                    nums[pointer] = kp.Key;
                    pointer++;
                }
            }

            return nums;
        }

        public static int[] EvenOddBit(int n)
        {
            var even = 0;
            var odd = 0;
            var number = 1;
            var i = 0;
            while (number <= n)
            {
                if ((number & n) >> i == 1)
                {
                    if (i % 2 == 0)
                        even++;
                    else
                        odd++;
                }

                number <<= 1;
                i++;
            }

            return new[] { even, odd };
        }

        public static string[] FindWords(string[] words)
        {
            var row1 = new HashSet<char>("qwertyuiop");
            var row2 = new HashSet<char>("asdfghjkl");
            var row3 = new HashSet<char>("zxcvbnm");
            var result = new List<string>(words.Length);
            foreach (var word in words)
            {
                if (new HashSet<char>(word.ToLower()).IsSubsetOf(row1)
                    || new HashSet<char>(word.ToLower()).IsSubsetOf(row2)
                    || new HashSet<char>(word.ToLower()).IsSubsetOf(row3))
                    result.Add(word);
            }

            return result.ToArray();
        }

        public int CountGoodSubstrings(string s)
        {
            var result = 0;
            var hashSet = new HashSet<int>(3);
            for (var i = 0; i < s.Length - 2; i++)
            {
                if (i > 0)
                    hashSet.Remove(s[i - 1]);
                hashSet.Add(s[i]);
                hashSet.Add(s[i + 1]);
                hashSet.Add(s[i + 2]);
                if (hashSet.Count == 3)
                    result++;
            }

            return result;
        }

        public int FindFinalValue(int[] nums, int original)
        {
            Array.Sort(nums);
            foreach (var x in nums)
            {
                if (x == original)
                {
                    original = 2 * original;
                }
            }

            return original;
        }

        public static IList<int> LuckyNumbers(int[][] matrix)
        {
            var minArrayElements = new List<(int X, int Y)>(matrix.Length);
            for (var y = 0; y < matrix.Length; y++)
            {
                var minX = int.MaxValue;
                var point = (-1, -1);
                for (var x = 0; x < matrix[0].Length; x++)
                {
                    if (matrix[y][x] <= minX)
                    {
                        point = (x, y);
                        minX = matrix[y][x];
                    }
                }

                minArrayElements.Add(point);
            }

            var result = new List<int>(matrix.Length);
            foreach (var point in minArrayElements)
            {
                var foundGreater = false;
                for (var y = 0; y < matrix.Length; y++)
                {
                    if (y == point.Y)
                        continue;
                    if (matrix[y][point.X] >= matrix[point.Y][point.X])
                    {
                        foundGreater = true;
                        break;
                    }
                }

                if (!foundGreater)
                    result.Add(matrix[point.X][point.Y]);
            }

            return result;
        }

        public int VowelStrings(string[] words, int left, int right)
        {
            return words
                .Select((w, i) => (w, i))
                .Where(x => x.i >= left)
                .Where(x => x.i <= left)
                .Where(x => new[] { 'a', 'e', 'i', 'o', 'u' }.Contains(x.w[0]))
                .Count(x => new[] { 'a', 'e', 'i', 'o', 'u' }.Contains(x.w[^1]));
        }

        public static string RemoveOccurrences(string line, string part)
        {
            while (true)
            {
                var index = line.IndexOf(part);
                if (index == -1)
                    break;
                line = line.Substring(0, index) + line.Substring(index + part.Length);
            }

            return line;
        }

        public int PercentageLetter(string line, char letter)
        {
            var result = 0.0;
            foreach (var x in line)
            {
                if (x == letter)
                    result += 1.0;
            }

            return (int)(result * 100) / line.Length;
        }

        public IList<string> SplitWordsBySeparator(IList<string> words, char separator)
        {
            var result = new List<string>();
            var sb = new StringBuilder();
            foreach (var word in words)
            {
                sb = new StringBuilder();
                foreach (var x in word)
                {
                    if (x == separator)
                    {
                        if (sb.Length > 0)
                        {
                            result.Add(sb.ToString());
                            sb = new StringBuilder();
                        }
                    }
                    else
                        sb.Append(x);
                }

                if (sb.Length > 0)
                    result.Add(sb.ToString());
            }

            return result;
        }

        public int SumOfEncryptedInt(int[] nums)
        {
            var sum = 0;
            foreach (var x in nums)
            {
                var current = x;
                var maxDigit = int.MinValue;
                var length = 0;
                while (current > 0)
                {
                    length++;
                    // maxDigit = int.MaxValue(current % 10, maxDigit);
                    current /= 10;
                }

                for (var i = 0; i < length; i++)
                    sum += (int)Math.Pow(10, i) * maxDigit;
            }

            return sum;
        }

        public static int MinDeletionSize(string[] lines)
        {
            var unsortedCount = 0;
            for (var x = 0; x < lines[0].Length; x++)
            {
                for (var y = 0; y < lines.Length - 1; y++)
                {
                    if (lines[y][x] > lines[y + 1][x])
                    {
                        unsortedCount++;
                        break;
                    }
                }
            }

            return unsortedCount;
        }

        public int CountTriples(int n)
        {
            var result = 0;
            for (var i = 1; i <= n; i++)
            for (var j = 1; j <= i; j++)
            for (var k = 1; k <= j; k++)
            {
                if (i * i + j * j == k * k)
                    result++;
            }

            return result;
        }

        public static string ReverseParentheses(string line)
        {
            var result = new StringBuilder(line);
            while (result.ToString().Contains('('))
            {
                var startIndex = 0;
                var buffer = new StringBuilder();
                for (var i = 0; i < line.Length; i++)
                {
                    if (result[i] == '\0')
                        continue;
                    if (result[i] == '(')
                    {
                        startIndex = i;

                        if (buffer.Length == 0)
                            continue;
                        buffer.Clear();
                    }
                    else if (result[i] == ')')
                    {
                        result[startIndex] = '\0';
                        result[i] = '\0';
                        for (var j = startIndex; j <= i; j++)
                        {
                            result[j] = '\0';
                        }

                        for (var j = 0; j < buffer.Length; j++)
                        {
                            result[startIndex + 1 + j] = buffer[buffer.Length - 1 - j];
                        }

                        buffer.Clear();
                        break;
                    }
                    else
                    {
                        buffer.Append(result[i]);
                    }
                }
            }

            var meow = new StringBuilder();
            for (var i = 0; i < result.Length; i++)
            {
                if (result[i] == '\0')
                    continue;
                meow.Append(result[i]);
            }

            return meow.ToString();
        }

        public int MinOperations1(string[] logs)
        {
            var depth = 0;
            foreach (var log in logs)
            {
                if (log == "./")
                    continue;
                if (log == "../")
                    depth = Math.Max(--depth, 0);
                else
                    depth++;
            }

            return depth;
        }

        public double AverageWaitingTime(int[][] customers)
        {
            var waiting = 0.0;
            var startTime = customers[0][0];
            foreach (var order in customers)
            {
                var arrivalTime = order[0];
                startTime = Math.Max(startTime, arrivalTime);
                waiting += (double)(startTime - arrivalTime + order[1]) / customers.Length;
                startTime += order[1];
            }

            return waiting;
        }

        public static int FindTheWinner(int n, int k)
        {
            var circle = new bool[n];
            var pointer = 0;
            var hashSet = Enumerable.Range(0, n).ToHashSet();
            while (hashSet.Count > 1)
            {
                var x = 0;
                while (circle[pointer])
                {
                    pointer++;
                    pointer %= n;
                }

                while (x != k - 1 || circle[pointer])
                {
                    if (!circle[pointer])
                    {
                        x++;
                    }

                    pointer++;
                    pointer %= n;
                }

                circle[pointer] = true;
                hashSet.Remove(pointer);
            }

            return hashSet.First() + 1;
        }

        public class ListNode
        {
            public int val { get; set; }
            public ListNode next { get; set; }

            public ListNode(int val = 0, ListNode next = null)
            {
                this.val = val;
                this.next = next;
            }
        }

        public static int NumWaterBottles(int bottles, int exchange)
        {
            var drunk = bottles;
            while (bottles >= exchange)
            {
                var fullBottles = bottles / exchange;
                drunk += fullBottles;
                bottles = fullBottles + bottles % exchange;
            }

            return drunk;
        }

        public static int PassThePillow(int n, int time)
        {
            var overflow = time % (n - 1);
            var done = time / (n - 1);
            if (done % 2 == 1)
                return n - overflow;
            return overflow + 1;
        }

        public int Fib(int n)
        {
            if (n == 0)
                return 0;
            if (n == 1)
                return 1;
            return Fib(n) + Fib(n - 1);
        }

        public int[] ReplaceElements(int[] arr)
        {
            var max = -1;
            for (var i = arr.Length - 1; i >= 0; i--)
            {
                var localMax = Math.Max(max, arr[i]);
                arr[i] = max;
                max = localMax;
            }

            return arr;
        }

        public static int FindMiddleIndex(int[] nums)
        {
            var leftSum = 0;
            var rightSum = nums.Sum();
            for (var i = 0; i < nums.Length; i++)
            {
                if (i > 0)
                    leftSum += nums[i - 1];
                rightSum -= nums[i];
                if (leftSum == rightSum)
                    return i;
            }

            return -1;
        }

        public int SplitNum(int num)
        {
            var ordered = num.ToString().ToCharArray();
            Array.Sort(ordered);
            var sb1 = new StringBuilder(ordered.Length / 2);
            var sb2 = new StringBuilder(ordered.Length / 2);
            for (var i = 0; i < ordered.Length; i++)
            {
                if (i % 2 == 0)
                    sb1.Append(ordered[i]);
                else
                    sb2.Append(ordered[i]);
            }

            return int.Parse(sb1.ToString()) + int.Parse(sb2.ToString());
        }

        public bool ThreeConsecutiveOdds(int[] arr)
        {
            if (arr.Length < 3)
                return false;
            for (var i = 1; i < arr.Length - 1; i++)
            {
                var index = i;
                var foundEven = arr[index - 1] % 2 == 0;
                if (arr[index] % 2 == 0)
                {
                    foundEven = true;
                    i++;
                }

                if (arr[index + 1] % 2 == 0)
                {
                    foundEven = true;
                    i++;
                }

                if (!foundEven)
                    return true;
            }

            return false;
        }

        public int[] SortArrayByParityII(int[] nums)
        {
            var result = new int[nums.Length];
            var lastEven = 1;
            var lastOdd = 0;
            for (var i = 0; i < nums.Length; i++)
            {
                if (nums[i] % 2 == 0)
                {
                    result[lastOdd] = nums[i];
                    lastOdd += 2;
                }
                else
                {
                    result[lastEven] = nums[i];
                    lastEven += 2;
                }
            }

            return result;
        }

        public IList<int> FindPeaks(int[] mountain)
        {
            var peaks = new List<int>();
            for (var i = 1; i < mountain.Length - 1; i++)
            {
                if (mountain[i] > mountain[i - 1] && mountain[i] > mountain[i + 1])
                {
                    peaks.Add(i);
                    i++;
                }
            }

            return peaks;
        }

        public static int CountSymmetricIntegers(int low, int high)
        {
            var result = 0;
            for (var x = low; x <= high; x++)
            {
                var number = x.ToString();
                if (number.Length % 2 == 1)
                {
                    x = (int)Math.Pow(10, number.Length);
                    continue;
                }

                var left = Enumerable.Range(0, number.Length / 2)
                    .Select(i => number[i] - '0')
                    .Sum();
                var right = Enumerable.Range(number.Length / 2, number.Length / 2)
                    .Select(i => number[i] - '0')
                    .Sum();

                if (left == right)
                    result++;
            }

            return result;
        }

        public int FindCenter(int[][] edges)
        {
            return edges
                .SelectMany(x => x)
                .GroupBy(x => x)
                .ToDictionary(x => x.Key, x => x.Count())
                .OrderByDescending(x => x.Value)
                .Select(x => x.Key)
                .First();
        }

        public int CountOperations(int num1, int num2)
        {
            var count = 0;
            while (num1 != 0 && num2 != 0)
            {
                if (num1 > num2)
                    (num1, num2) = (num2, num1);
                num2 -= num1;
                count++;
            }

            return count;
        }

        public int SumOfSquares(int[] nums)
        {
            var squares = 1;
            for (var i = 0; i < nums.Length; i++)
            {
                if (nums.Length % i == 0)
                    squares += nums[i] * nums[i];
            }

            return squares;
        }

        public int MinOperations(int n)
        {
            var result = 1;
            for (var i = 0; i < n / 2; i++)
            {
                result += 2 * (i + 1) * result;
            }

            return result;
        }

        public bool IsArraySpecial1(int[] nums)
        {
            for (var i = 0; i < nums.Length - 1; i++)
                if (nums[i] % 2 + nums[i + 1] % 2 != 1)
                    return false;

            return true;
        }

        public int[] SortArrayByParity(int[] nums)
        {
            var result = new int[nums.Length];
            var left = 0;
            var right = nums.Length - 1;
            for (var i = 0; i < nums.Length; i++)
            {
                if (nums[i] % 2 == 0)
                    result[left++] = nums[i];
                else
                    result[right--] = nums[i];
            }

            return result;
        }

        public int FindNumbers(int[] nums)
        {
            var result = 0;
            foreach (var x in nums)
            {
                var length = 0;
                var current = x;
                while (current > 0)
                {
                    length++;
                    current /= 10;
                }

                if (length % 2 == 0)
                    result++;
            }

            return result;
        }

        public int CountCompleteDayPairs(int[] hours)
        {
            var result = 0;
            for (var i = 0; i < hours.Length; i++)
            for (var j = i + 1; j < hours.Length; j++)
            {
                if (hours[i] + hours[j] > 0 && hours[i] + hours[j] % 24 == 0)
                    result++;
            }

            return result;
        }

        public int MinSteps(string s, string t)
        {
            var dict1 = s
                .GroupBy(x => x)
                .ToDictionary(x => x.Key, x => x.Count());
            var dict2 = t
                .GroupBy(x => x)
                .ToDictionary(x => x.Key, x => x.Count());
            var steps = 0;
            foreach (var kp in dict2)
            {
                if (!dict1.ContainsKey(kp.Key))
                {
                }
                else
                {
                    steps += Math.Abs(dict1[kp.Key] - kp.Value);
                }
            }

            return steps;
        }

        public static bool JudgeSquareSum(int c)
        {
            var x = 0L;
            var y = (long)(Math.Sqrt(c) + 1);
            while (x <= y)
            {
                if (x * x + y * y == c)
                    return true;
                if (x * x + y * y > c)
                    y--;
                else
                    x++;
            }

            return false;
        }

        public string SortVowels(string line)
        {
            var sb = new StringBuilder(line);
            var vowels = new List<char>();
            var indexes = new List<int>();
            for (var i = 0; i < line.Length; i++)
            {
                if (line[i] == 'a' || line[i] == 'A' ||
                    line[i] == 'e' || line[i] == 'E' ||
                    line[i] == 'i' || line[i] == 'I' ||
                    line[i] == 'o' || line[i] == 'O' ||
                    line[i] == 'u' || line[i] == 'U')
                {
                    vowels.Add(line[i]);
                    indexes.Add(i);
                }
            }

            vowels.Sort();
            for (var i = 0; i < vowels.Count; i++)
                sb[indexes[i]] = vowels[i];

            return sb.ToString();
        }

        public static int MaxProfitAssignment(int[] difficulty, int[] profit, int[] worker)
        {
            var meow = difficulty.Select((d, i) => (d, profit[i]))
                .OrderBy(x => x.d)
                .ThenBy(x => x.Item2)
                .ToArray();
            Array.Sort(worker);
            var pointer = worker.Length - 1;
            var result = 0;
            var i = difficulty.Length - 1;
            while (i >= 0 && pointer >= 0)
            {
                if (worker[pointer] >= meow[i].d)
                {
                    result += meow[i].Item2;
                    pointer--;
                }
                else
                {
                    i--;
                }
            }

            return result;
        }

        public static int MaximumNumberOfStringPairs(string[] words)
        {
            var visited = new HashSet<(char a, char b)>();
            var counter = 0;
            foreach (var x in words)
                if (visited.Contains((x[1], x[0])))
                    counter++;
                else
                    visited.Add(((x[0], x[1])));

            return counter;
        }

        public static int CountSeniors(string[] details)
        {
            return details.Count(x => x[11] > '6' || (x[11] == '6' && x[12] > '0'));
        }

        public char FindTheDifference(string source, string target)
        {
            var dict1 = source
                .GroupBy(x => x)
                .ToDictionary(x => x.Key, x => x.Count());
            var dict2 = target
                .GroupBy(x => x)
                .ToDictionary(x => x.Key, x => x.Count());
            foreach (var kp in dict2)
            {
                if (!dict1.ContainsKey(kp.Key))
                    return kp.Key;
                if (dict1.ContainsKey(kp.Key) && dict1[kp.Key] != kp.Value)
                    return kp.Key;
            }

            return '\0';
        }

        public int MinMovesToSeat(int[] seats, int[] students)
        {
            Array.Sort(seats);
            Array.Sort(students);
            var result = 0;
            for (var i = 0; i < seats.Length; i++)
                result += Math.Abs(seats[i] - students[i]);

            return result;
        }

        public void SortColors(int[] nums)
        {
            for (var i = 0; i < nums.Length; i++)
            for (var j = i + 1; j < nums.Length; j++)
                if (nums[j] < nums[i])
                    (nums[i], nums[j]) = (nums[j], nums[i]);
        }

        public static string ReverseVowels(string line)
        {
            var vowels = new HashSet<char> { 'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U' };
            var result = new char[line.Length];
            var indexes = new List<int>();
            for (var i = 0; i < line.Length; i++)
            {
                if (vowels.Contains(line[i]))
                    indexes.Add(i);
            }

            var pointer = 0;
            for (var i = 0; i < line.Length; i++)
            {
                if (pointer < indexes.Count && i == indexes[pointer])
                {
                    result[i] = line[indexes.Count - 1 - pointer];
                    pointer++;
                }
                else
                    result[i] = line[i];
            }

            return new string(result);
        }

        public int[] RelativeSortArray(int[] arr1, int[] arr2)
        {
            var order = arr2
                .Select((x, i) => (x, i))
                .ToDictionary(x => x.x, x => x.i);
            var bastards = new List<int>(arr1.Length);
            var result = new List<int>(arr1.Length);
            foreach (var x in arr1)
                if (!order.ContainsKey(x))
                    bastards.Add(x);
                else
                    result.Add(x);
            result = result
                .OrderBy(x => order[x])
                .ToList();
            result.AddRange(bastards.OrderBy(x => x));
            return result.ToArray();
        }

        public int HeightChecker(int[] heights)
        {
            var sortedCopy = heights
                .OrderBy(x => x)
                .ToArray();
            return Enumerable.Range(0, heights.Length)
                .Count(i => sortedCopy[i] != heights[i]);
        }

        public int[] Intersect(int[] nums1, int[] nums2)
        {
            var result = new List<int>();
            var dict1 = nums1
                .GroupBy(x => x)
                .ToDictionary(x => x.Key, x => x.Count());
            foreach (var x in nums2)
            {
                if (dict1.ContainsKey(x) && dict1[x] > 0)
                {
                    result.Add(x);
                    dict1[x]--;
                }
            }

            return result.ToArray();
        }

        public void MoveZeroes(int[] nums)
        {
            var withoutZeroes = new List<int>();
            foreach (var x in nums)
            {
                if (x != 0)
                    withoutZeroes.Add(x);
            }

            for (var i = 0; i < nums.Length; i++)
            {
                if (i < withoutZeroes.Count)
                    nums[i] = withoutZeroes[i];
                else
                {
                    nums[i] = 0;
                }
            }
        }

        public static string ReplaceWords(IList<string> dictionary, string sentence)
        {
            var result = new List<string>();
            var words = sentence.Split(" ");
            foreach (var word in words)
            {
                string replace = null;
                foreach (var root in dictionary)
                {
                    if (!word.StartsWith(root))
                        continue;
                    if (replace == null)
                        replace = root;
                    else if (root.Length < replace.Length)
                        replace = root;
                }

                result.Add(replace ?? word);
            }

            return string.Join(" ", result);
        }

        public string[] SortPeople(string[] names, int[] heights)
        {
            return heights
                .Select((h, i) => (h, names[i]))
                .OrderByDescending(x => x.h)
                .Select(x => x.Item2)
                .ToArray();
        }

        public static IList<string> CommonChars(string[] words)
        {
            if (words.Length == 0)
                return Array.Empty<string>();
            var buffer = words[0]
                .GroupBy(x => x)
                .ToDictionary(x => x.Key.ToString(), x => x.Count());
            foreach (var word in words.Skip(1))
            {
                var current = word
                    .GroupBy(x => x)
                    .ToDictionary(x => x.Key.ToString(), x => x.Count());
                foreach (var x in buffer.Keys)
                    if (current.TryGetValue(x, out var value))
                        buffer[x] = Math.Min(buffer[x], value);
                    else
                        buffer.Remove(x);
            }

            var result = new List<string>();
            foreach (var kp in buffer)
                for (var i = 0; i < kp.Value; i++)
                    result.Add(kp.Key);

            return result;
        }

        public int MinimumChairs(string line)
        {
            var people = 0;
            var maxVisitors = 0;
            foreach (var x in line)
            {
                if (x == 'E')
                    people++;
                else if (x == 'L')
                    people--;
                maxVisitors = Math.Max(maxVisitors, people);
            }

            return maxVisitors;
        }

        public static int LongestPalindrome(string line)
        {
            var buffer = new HashSet<char>();
            var length = 0;
            foreach (var x in line.Where(x => !buffer.Add(x)))
            {
                length += 2;
                buffer.Remove(x);
            }

            return length + (buffer.Count == 0 ? 0 : 1);
        }

        public int CountTestedDevices(int[] batteryPercentages)
        {
            var testedDevices = 0;
            var dx = 0;
            foreach (var x in batteryPercentages)
            {
                if (x - dx <= 0)
                    continue;
                dx++;
                testedDevices++;
            }

            return testedDevices;
        }

        public static int CountKeyChanges(string line)
        {
            var dx = 'a' - 'A';
            var count = 0;
            for (var i = 0; i < line.Length - 1; i++)
            {
                if (Math.Abs(line[i] - line[i + 1]) != dx && line[i] - line[i + 1] != 0)
                    count++;
            }

            return count;
        }

        public int AppendCharacters(string source, string target)
        {
            var substring = new StringBuilder(target.Length);
            var lastIndex = -1;
            var subIndex = -1;
            for (var i = 0; i < target.Length; i++)
            {
                substring.Append(target[i]);
                var index = source.IndexOf(substring.ToString(), StringComparison.Ordinal);
                if (index != -1)
                    subIndex = index;
                else
                    lastIndex = i;
            }

            var length = subIndex - lastIndex;

            // substring = new StringBuilder();
            // for (var i = lastIndex; i < target.Length; i++)
            // {
            //     substring.Append(target[i]);
            //     var index = source.IndexOf(substring.ToString(), subIndex);
            //     if (index != -1)
            //         subIndex = index;
            //     else
            //         lastIndex = i;
            // }
            //
            return target.Substring(lastIndex).Length;
        }

        public int ScoreOfString(string line)
        {
            var result = 0;
            for (var i = 0; i < line.Length - 1; i++)
                result = Math.Abs(line[i] - line[i + 1]);
            return result;
        }

        public static int[] SingleNumber(int[] nums)
        {
            return nums
                .GroupBy(x => x)
                .Where(x => x.Count() == 2)
                .Select(x => x.Key)
                .ToArray();
        }

        public int CountAsterisks(string line)
        {
            var walls = 0;
            var stars = 0;
            foreach (var x in line)
                if (x == '|')
                    walls++;
                else if (x == '*' && walls % 2 == 0)
                    stars++;

            return stars;
        }

        public static int NumSteps(string binary)
        {
            var meow = binary.ToCharArray();
            var count = 0;
            var pointer = binary.Length - 1;
            var increments = 0;
            while (pointer != 0)
            {
                if (meow[pointer] == '0' && increments > 0)
                {
                    meow[pointer] = '1';
                    increments--;
                }

                if (meow[pointer] == '1')
                {
                    increments++;
                    count++;
                }
                else
                {
                    if (increments > 0)
                        increments--;
                    else
                        count++;
                }

                if (pointer > 0)
                    pointer--;
            }

            return count + increments + 1;
        }

        public string ReplaceDigits(string line)
        {
            var sb = new StringBuilder(line.Length);
            for (var i = 0; i < line.Length - 1; i += 2)
            {
                sb.Append(line[i]);
                sb.Append((char)(line[i] + line[i + 1] - '0'));
            }

            return sb.ToString();
        }

        public int[] FindIntersectionValues(int[] nums1, int[] nums2)
        {
            var hashSet1 = nums1.ToHashSet();
            var hashSet2 = nums2.ToHashSet();
            return new[]
            {
                nums1.Count(x => hashSet2.Contains(x)),
                nums2.Count(x => hashSet1.Contains(x))
            };
        }

        public int MaxProductDifference(int[] nums)
        {
            Array.Sort(nums);
            return nums[^1] * nums[^2] - nums[0] * nums[1];
        }

        public int SpecialArray(int[] nums)
        {
            for (var i = 0; i < nums.Length; i++)
            {
                var rights = nums.Length - i;
                if (nums[i] >= rights)
                    return rights;
            }

            return -1;
        }

        public int MaxProduct(int[] nums)
        {
            var max1 = nums.Max();
            var max2 = 0;
            foreach (var x in nums)
            {
                if (x > max2 && x != max1)
                    max1 = x;
            }

            return (max1 - 1) * (max2 - 1);
        }

        public int SumOfTheDigitsOfHarshadNumber(int x)
        {
            var n = x;
            var sum = 0;
            while (n > 0)
            {
                sum += n % 10;
                n /= 10;
            }

            return x % sum == 0 ? sum : -1;
        }

        public int UniqueMorseRepresentations(string[] words)
        {
            var morse = new[]
            {
                ".-", "-...", "-.-.", "-..", ".", "..-.", "--.", "....", "..", ".---", "-.-", ".-..", "--", "-.", "---",
                ".--.", "--.-", ".-.", "...", "-", "..-", "...-", ".--", "-..-", "-.--", "--.."
            };
            var hashSet = words
                .Select(word => string.Join("", word.Select(x => morse[x - 'a'])))
                .ToHashSet();
            return hashSet.Count;
        }

        public int NumberOfPairs(int[] nums1, int[] nums2, int k)
        {
            var count = 0;
            for (var i = 0; i < nums1.Length; i++)
            for (var j = 0; i < nums2.Length; j++)
                if (nums1[i] % (nums2[j] * k) == 0)
                    count++;
            return count;
        }

        public string ReverseWords(string line)
        {
            var words = new List<List<char>>();
            foreach (var x in line)
            {
                if (x == ' ')
                {
                    words.Add(new List<char>());
                }
                else
                {
                    if (words.Count == 0)
                    {
                        words.Add(new List<char>());
                    }

                    words[^1].Add(x);
                }
            }

            var sb = new StringBuilder();
            for (var j = 0; j < words.Count; j++)
            {
                for (var i = 0; i < words[j].Count; i++)
                {
                    sb.Append(words[j][words[j].Count - 1 - i]);
                }

                if (j == words.Count - 1)
                    break;
                sb.Append(' ');
            }

            return sb.ToString();
        }

        public int LargestAltitude(int[] gain)
        {
            var height = 0;
            var maxHeight = 0;
            foreach (var dh in gain)
            {
                height += dh;
                maxHeight = Math.Max(maxHeight, height);
            }

            return maxHeight;
        }

        private static string[] excluded = { "1", "2", "3" };

        public int NumberOfMatches(int teams)
        {
            var count = 0;
            while (teams > 0)
            {
                var matches = teams / 2;
                teams -= matches;
                count += matches;
            }

            return count;
        }

        public int CountDigits(int value)
        {
            var count = 0;
            var n = value;
            while (n != 0)
            {
                if (value % (n % 10) == 0)
                    count++;
                n /= 10;
            }

            return count;
        }

        public int[] NumberGame(int[] nums)
        {
            Array.Sort(nums);
            var result = new int[nums.Length];
            for (var i = 0; i < nums.Length; i += 2)
            {
                result[i] = nums[i + 1];
                result[i + 1] = nums[i];
            }

            return result;
        }

        // public int[] MinOperations(string boxes)
        // {
        //     var result = new int[boxes.Length];
        //     for (var i = 0; i < boxes.Length; i++)
        //     for (var j = 0; j < boxes.Length; j++)
        //     {
        //         if (i == j)
        //             continue;
        //         result[i] += Math.Abs(i - j) * j;
        //     }
        //
        //     return result;
        // }

        public int[] SmallerNumbersThanCurrent(int[] nums)
        {
            var sorted = nums
                .OrderBy(x => x)
                .Select((value, index) => (value, index))
                .GroupBy(x => x.value)
                .ToDictionary(x => x.Key,
                    x => x.Min(y => y.index));

            return nums.Select(x => sorted[x]).ToArray();
        }

        public IList<bool> KidsWithCandies(int[] candies, int extraCandies)
        {
            var max = candies.Max();
            return candies.Select(x => x + extraCandies >= max).ToArray();
        }

        public int CountPairs(IList<int> nums, int target)
        {
            nums = nums.OrderBy(x => x).ToArray();
            var count = 0;
            for (var i = 0; i < nums.Count - 1; i++)
            {
                for (var j = i + 1; j < nums.Count; j++)
                {
                    if (nums[i] + nums[j] >= target)
                        break;
                    count++;
                }
            }

            return count;
        }

        public string Interpret(string command)
        {
            var sb = new StringBuilder(command.Length);
            for (var i = 0; i < command.Length; i++)
            {
                if (command[i] == '(')
                {
                    if (command[i + 1] == 'a')
                    {
                        sb.Append("al");
                        i += 3;
                    }
                    else
                    {
                        sb.Append("o");
                        i += 1;
                    }
                }
                else
                    sb.Append(command[i]);
            }

            return sb.ToString();
        }

        public int MaximumWealth(int[][] accounts)
        {
            return accounts.Select(acc => acc.Sum()).Prepend(0).Max();
        }

        public static int ArithmeticTriplets(int[] nums, int diff)
        {
            var buffer = nums.ToHashSet();
            var count = 0;
            foreach (var x in nums)
            {
                if (buffer.Contains(x - diff))
                    count++;
            }

            return count;
        }

        public int MinBitFlips(int start, int goal)
        {
            var xor = start ^ goal;
            var count = 0;
            for (var i = 0; i < 31; i++)
            {
                if ((xor & (1 << i)) >> i == 1)
                    count++;
            }

            return count;
        }

        public string ToLowerCase(string line)
        {
            var sb = new StringBuilder(line.Length);
            foreach (var x in line)
            {
                if (x >= 'A' && x <= 'Z')
                    sb.Append((char)(x + 'a' - 'A'));
                else
                {
                    sb.Append(x);
                }
            }

            return sb.ToString();
        }

        public static bool UniqueOccurrences(int[] nums)
        {
            Array.Sort(nums);
            var occurrences = new HashSet<int>();
            var current = nums[0];
            for (var i = 0; i < nums.Length; i++)
            {
                var currentOccurences = 0;
                while (i < nums.Length && nums[i] == current)
                {
                    currentOccurences++;
                    i++;
                }

                if (!occurrences.Add(currentOccurences))
                    return false;
                if (i == nums.Length)
                    return true;
                current = nums[i];
                i--;
            }

            return true;
        }

        public int[] DistinctDifferenceArray(int[] nums)
        {
            var suffix = nums
                .GroupBy(x => x)
                .ToDictionary(x => x.Key, x => x.Count());
            var prefix = new HashSet<int>();
            for (var i = 0; i < nums.Length; i++)
            {
                prefix.Add(nums[i]);
                suffix[nums[i]]--;
                if (suffix[nums[i]] == 0)
                    suffix.Remove(nums[i]);

                nums[i] = prefix.Count - suffix.Keys.Count;
            }

            return nums;
        }

        public int SumBase(int n, int k)
        {
            var sum = 0;
            while (n > 0)
            {
                sum += n % k;
                n /= k;
            }

            return sum;
        }

        public static long MaximumHappinessSum(int[] happiness, int k)
        {
            Array.Sort(happiness);
            var sum = 0;
            for (var i = 0; i < k; i++)
            {
                sum += happiness[happiness.Length - 1 - i] - i;
                happiness[happiness.Length - 1 - i] = 0;
            }

            for (var i = 0; i < happiness.Length - k; i++)
            {
                if (happiness[i] - k <= 0)
                    continue;
                sum += happiness[i] - k;
            }

            return sum;
        }

        public static int PrefixCount(string[] words, string pref)
        {
            return words
                .Count(x => x.StartsWith(pref));
            // var visited = new HashSet<string>();
            // for (var pointer = 0; pointer < pref.Length; pointer++)
            // {
            //     if (visited.Count == words.Length)
            //         break;
            //     foreach (var word in words)
            //     {
            //         if (visited.Contains(word))
            //             continue;
            //         
            //         if (word.Length < pref.Length || word[pointer] != pref[pointer])
            //             visited.Add(word);
            //     }
            // }
            // return words.Length - visited.Count;
        }

        public static string[] FindRelativeRanks(int[] score)
        {
            var r = score
                .Select((x, i) => (x, i))
                .OrderByDescending(y => y.x)
                .Select(x => x.i);
            var result = new string[score.Length];
            var i = 0;
            foreach (var x in r)
            {
                if (i == 0)
                    result[x] = "Gold Medal";
                else if (i == 1)
                    result[x] = "Silver Medal";
                else if (i == 2)
                    result[x] = "Bronze Medal";
                else result[x] = (i + 1).ToString();
                i++;
            }

            return result;
        }

        public int[] SortByBits(int[] arr)
        {
            return arr
                .Select(x =>
                {
                    var count = 0;
                    var power = 0;
                    while (power <= 31)
                    {
                        if ((x & (1 << power)) == 1)
                            count++;
                        power++;
                    }

                    return new { Number = x, Count = count };
                })
                .OrderBy(x => x.Count)
                .ThenBy(x => x.Number)
                .Select(x => x.Number)
                .ToArray();
        }

        public static IList<int> SelfDividingNumbers(int left, int right)
        {
            var result = new List<int>();
            for (var i = left; i <= right; i++)
            {
                var meow = i;
                while (meow > 0)
                {
                    if (meow % 10 == 0 || i % (meow % 10) != 0)
                        break;
                    meow /= 10;
                }

                if (meow == 0)
                    result.Add(i);
            }

            return result;
        }

        public static int CompareVersion(string version1, string version2)
        {
            var buffer1 = version1.Split('.');
            var buffer2 = version2.Split('.');

            for (var i = 0; i < Math.Min(buffer1.Length, buffer2.Length); i++)
            {
                var number1 = int.Parse(buffer1[i]);
                var number2 = int.Parse(buffer2[i]);
                if (number1 < number2)
                    return -1;
                if (number1 > number2)
                    return 1;
            }

            if (buffer1.Length > buffer2.Length)
            {
                for (var i = buffer2.Length; i < buffer1.Length; i++)
                    if (int.Parse(buffer1[i]) > 0)
                        return 1;
            }

            if (buffer2.Length > buffer1.Length)
            {
                for (var i = buffer1.Length; i < buffer2.Length; i++)
                    if (int.Parse(buffer2[i]) > 0)
                        return -1;
            }

            return 0;
        }
    }
}