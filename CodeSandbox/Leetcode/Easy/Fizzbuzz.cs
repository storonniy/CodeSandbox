using System.Collections;
using System.Diagnostics.Contracts;
using System.Text;
using CodeSandbox.Leetcode.Medium;

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

    public static int MinimumLength(string s)
    {
        var count = s.Length;
        var left = 0;
        var right = s.Length - 1;
        while (left < right)
        {
            if (s[left] != s[right])
                break;
            var symbol = s[left];
            while (left < right && s[left] == symbol)
            {
                left++;
                count--;
            }

            while (left < right && s[right] == symbol)
            {
                right--;
                count--;
            }
        }

        return count;
    }

    public bool HasCycle(RangeBitwiseAnd1.Solution.ListNode head)
    {
        var slow = head;
        var fast = head;
        while (slow != null && fast != null && fast.next != null)
        {
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast)
                return true;
        }

        return false;
    }

    public RangeBitwiseAnd1.Solution.ListNode MiddleNode(RangeBitwiseAnd1.Solution.ListNode head)
    {
        var root = head;
        var count = 0;
        while (root != null)
        {
            count++;
            root = root.next;
        }

        for (var i = 0; i < count / 2; i++)
            head = head.next;
        return head;
    }

    public int MaxFrequencyElements(int[] nums)
    {
        var freq = new Dictionary<int, int>();
        var maxFreq = 0;
        foreach (var n in nums)
        {
            if (!freq.ContainsKey(n))
                freq.Add(n, 0);
            freq[n]++;
            maxFreq = Math.Max(maxFreq, freq[n]);
        }

        return freq.Values.Where(x => x == maxFreq).Sum();
    }

    public static string Multiply(string num1, string num2)
    {
        var n1 = num1.Reverse().ToArray();
        var n2 = num2.Reverse().ToArray();
        var a = new List<StringBuilder>();
        var maxLength = 0;
        var remainder = 0;
        foreach (var x in n1)
        {
            var sb = new StringBuilder();
            remainder = 0;
            for (var j = 0; j < n2.Length; j++)
            {
                var y = n2[j];
                var xy = (x - '0') * (y - '0') + remainder;
                var digit = xy % 10;
                remainder = xy / 10;
                sb.Append(digit);
                if (j == n2.Length - 1)
                {
                    if (remainder > 0)
                        sb.Append(remainder);
                    a.Add(sb);
                    maxLength = Math.Max(maxLength, sb.Length);
                }
            }
        }

        if (a.Count == 1)
            return new string(a[0].ToString().Reverse().ToArray());

        for (var i = 0; i < a.Count; i++)
        {
            for (var j = 0; j < i; j++)
            {
                a[i].Insert(0, '0');
                maxLength = Math.Max(maxLength, a[i].Length);
            }
        }

        foreach (var t in a)
        {
            for (var j = 0; j < maxLength - t.Length; j++)
                t.Append('0');
        }

        var result = new StringBuilder();

        for (var i = 0; i < maxLength; i++)
        {
            var xy = a.Select(x => x[i] - '0').Sum() + remainder;
            var digit = xy % 10;
            remainder = xy / 10;
            result.Append(digit);
            if (i == maxLength - 1)
            {
                if (remainder > 0)
                    result.Append(remainder);
            }
        }

        return new string(result.ToString().Reverse().ToArray());
    }

    public int[] LeftRightDifference(int[] nums)
    {
        var leftSum = new int[nums.Length];
        var rightSum = new int[nums.Length];
        var left = 0;
        var right = 0;
        for (var i = 0; i < nums.Length; i++)
        {
            if (i == 0)
            {
                left += nums[i];
                right += nums[nums.Length - 1 - i];
                continue;
            }

            leftSum[i] = left;
            rightSum[nums.Length - 1 - i] = right;
            left += nums[i];
            right += nums[nums.Length - 1 - i];
        }

        var result = new int[nums.Length];
        for (var i = 0; i < nums.Length; i++)
        {
            var x = leftSum[i] - rightSum[i];
            result[i] = x < 0 ? -x : x;
        }

        return result;
    }

    public int PivotInteger(int n)
    {
        var squaredPivot = (n * n + n) / 2;
        var x = (int)Math.Sqrt(squaredPivot);
        return squaredPivot == x * x ? x : -1;
    }

    public int[] ProductExceptSelf(int[] nums)
    {
        var product = 1;
        var indexOfFirstZero = -1;
        for (var i = 0; i < nums.Length; i++)
        {
            if (nums[i] != 0)
            {
                product *= nums[i];
                continue;
            }

            if (indexOfFirstZero != -1)
                return new int[nums.Length];

            indexOfFirstZero = i;
        }

        if (indexOfFirstZero != -1)
        {
            var result = new int[nums.Length];
            result[indexOfFirstZero] = product;
            return result;
        }

        var leftProduct = new int[nums.Length];
        leftProduct[0] = 1;
        var rightProduct = new int[nums.Length];
        rightProduct[^1] = 1;

        for (var i = 1; i < nums.Length; i++)
            leftProduct[i] = leftProduct[i - 1] * nums[i - 1];

        for (var i = nums.Length - 2; i >= 0; i--)
            rightProduct[i] = rightProduct[i + 1] * nums[i + 1];

        for (var i = 0; i < nums.Length; i++)
            nums[i] = leftProduct[i] * rightProduct[i];

        return nums;
    }

    // public static int[] SmallerNumbersThanCurrent(int[] nums)
    // {
    //     var copy = nums
    //         .OrderBy(x => x)
    //         .ToArray();
    //     var counts = nums
    //         .GroupBy(x => x)
    //         .ToDictionary(x => x.Key, x => x.Count());
    //     var result = new int[nums.Length];
    //     var lefts = new int[nums.Length];
    //     for (var i = 0; i < nums.Length; i++)
    //     {
    //         lefts[i] = copy[]
    //     }
    //     for (var i = 1; i < nums.Length; i++)
    //     {
    //         result[i] = lefts[i];
    //     }
    //
    //     return result;
    // }

    public int NumberOfSteps(int number)
    {
        var count = 0;
        while (number > 0)
        {
            if (number % 2 == 0)
                number /= 2;
            else
                number--;
            count++;
        }

        return count;
    }

    public int CountMatches(IList<IList<string>> items, string ruleKey, string ruleValue)
    {
        var ruleIndex = GetItemIndexByName(ruleKey);
        return items
            .Count(x => x[ruleIndex] == ruleValue);
    }

    int GetItemIndexByName(string name)
    {
        switch (name)
        {
            case "type":
                return 0;
            case "color":
                return 1;
            case "name":
                return 2;
            default:
                throw new ArgumentException();
        }
    }

    public static int FirstMissingPositive(int[] nums)
    {
        var n = nums.Length;
        var min = nums.Min();
        var threshold = n;
        if (min < 1)
            threshold = n - 1;
        for (var i = 0; i < n; i++)
        {
            if (nums[i] < 1 || nums[i] > threshold)
                nums[i] = 0;
        }

        var tmp = -1;
        for (var i = 0; i < n; i++)
        {
            if (nums[i] == i + 1)
                continue;
            if (nums[i] < 1 || nums[i] > n)
                continue;
            tmp = nums[nums[i] - 1];
            nums[nums[i] - 1] = nums[i];
            if (tmp < 1 || tmp > n)
                continue;
            nums[tmp - 1] = tmp;
        }

        for (var i = 0; i < n; i++)
            if (nums[i] != i + 1)
                return i + 1;
        return n + 1;
    }

    public int NumSubarrayProductLessThanK(int[] nums, int k)
    {
        var count = 0;
        var tail = 0;
        var product = 0;
        for (var i = 0; i < nums.Length; i++)
        {
            product *= nums[i];
            while (product >= k && tail <= i)
            {
                product /= nums[tail];
                tail++;
            }

            count += i - tail + 1;
        }

        return count;
    }

    public static bool ArrayStringsAreEqual(string[] word1, string[] word2)
    {
        return string.Join("", word1) == string.Join("", word2);
        var partPointer1 = 0;
        var partPointer2 = 0;
        var symbolPointer1 = 0;
        var symbolPointer2 = 0;
        while (partPointer1 < word1.Length
               && partPointer2 < word2.Length
               && symbolPointer1 < word1[partPointer1].Length
               && symbolPointer2 < word2[partPointer2].Length)
        {
            if (word1[partPointer1][symbolPointer1] != word2[partPointer2][symbolPointer2])
                return false;
            if (symbolPointer1 == word1[partPointer1].Length - 1)
            {
                symbolPointer1 = 0;
                partPointer1++;
            }
            else
                symbolPointer1++;

            if (symbolPointer2 == word2[partPointer2].Length - 1)
            {
                symbolPointer2 = 0;
                partPointer2++;
            }
            else
                symbolPointer2++;
        }

        return partPointer1 == word1.Length
               && symbolPointer1 == 0
               && partPointer2 == word2.Length
               && symbolPointer2 == 0;
    }

    public static int SumOddLengthSubarrays(int[] arr)
    {
        var result = 0;
        var n = arr.Length;

        var sum = 0;
        var sumByIndex = new Dictionary<int, int>();
        for (var i = 0; i < n; i++)
        {
            sum += arr[i];
            sumByIndex.Add(i, sum);
        }

        for (var right = 0; right < n; right++)
        {
            sum = sumByIndex[right];
            for (var left = 0; left <= right; left++)
            {
                if ((right - left + 1) % 2 == 0)
                {
                    sum -= arr[left];
                    result += sum;
                }
            }
        }

        return result;
    }

    public bool CheckIfPangram(string sentence)
    {
        var buffer = new HashSet<char>();
        foreach (var x in sentence)
        {
            if (!buffer.Contains(x))
                buffer.Add(x);
        }

        return buffer.Count == 26;
    }

    public int MaxSubarrayLength(int[] nums, int k)
    {
        var n = nums.Length;
        var dict = new Dictionary<int, int>();
        var maxLength = 0;
        var left = 0;
        for (var i = 0; i < n; i++)
        {
            dict[nums[i]] = dict.GetValueOrDefault(nums[i], 0) + 1;
            while (dict[nums[i]] > k)
            {
                dict[nums[left++]]--;
                maxLength = Math.Max(maxLength, i - left + 1);
            }
        }

        return maxLength;
    }

    public class Solution
    {
        public int MaxSubarrayLength(int[] nums, int k)
        {
            var n = nums.Length;
            var dict = new HashSet<int>();
            var result = 0;
            var left = 0;
            for (var i = 0; i < n; i++)
            {
                dict.Add(nums[i]);
                while (dict.Count > k)
                {
                    result++;
                    dict.Remove(nums[left++]);
                }
            }

            return result;
        }
    }

    public long CountSubarrays(int[] nums, int k)
    {
        var max = nums.Max();
        var count = 0;
        var result = 0L;
        var left = 0;
        for (var right = 0; right < nums.Length; right++)
        {
            if (nums[right] == max)
                count++;
            while (count >= k)
            {
                result += nums.Length - right;
                if (nums[left] == max)
                    count--;
                left++;
            }
        }

        return result;
    }

    public long CountSubarrays(int[] nums, int min, int max)
    {
        var result = 0L;

        var left = -1;
        var right = -1;
        var lastBomb = -1;
        for (var i = 0; i < nums.Length; i++)
        {
            if (nums[i] < min || nums[i] > max)
                lastBomb = i;
            if (nums[i] == min)
                left = i;
            if (nums[i] == max)
                right = i;
            var range = Math.Min(left, right) - lastBomb;
            if (range > 0)
                result += range;
        }

        return result;
    }

    public int LengthOfLastWord(string line)
    {
        var length = 0;
        var pointer = line.Length - 1;
        while (pointer >= 0 && line[pointer] == ' ')
        {
            pointer--;
        }

        while (pointer >= 0 && line[pointer] != ' ')
        {
            length++;
            pointer--;
        }

        return length;
    }

    public static bool IsIsomorphic(string line1, string line2)
    {
        var sb1 = new StringBuilder(line1.Length);
        var sb2 = new StringBuilder(line1.Length);
        var dict1 = new Dictionary<char, char>();
        var dict2 = new Dictionary<char, char>();
        for (var i = 0; i < line1.Length; i++)
        {
            if (!dict1.ContainsKey(line1[i]))
                dict1.Add(line1[i], (char)(i + 'a'));
            sb1.Append(dict1[line1[i]]);
            if (!dict2.ContainsKey(line2[i]))
                dict2.Add(line2[i], (char)(i + 'a'));
            sb2.Append(dict2[line2[i]]);
        }

        return sb1.ToString() == sb2.ToString();
    }

    public static bool Exist(char[][] board, string word)
    {
        var width = board[0].Length;
        var height = board.Length;

        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                if (Find(board, word, 1, x, y))
                    return true;
            }
        }

        return false;
    }

    private static bool Find(char[][] board, string word, int pointer, int x, int y)
    {
        var width = board[0].Length;
        var height = board.Length;
        if (pointer == word.Length)
            return true;
        if (x < 0 || y < 0 || x > width - 1 || y > height - 1)
            return false;
        if (board[y][x] != word[pointer])
            return false;
        var meow = board[y][x];
        board[y][x] = 'x';
        var r = Find(board, word, pointer + 1, x + 1, y) ||
                Find(board, word, pointer + 1, x - 1, y) ||
                Find(board, word, pointer + 1, x, y + 1) ||
                Find(board, word, pointer + 1, x, y - 1);
        board[y][x] = meow;
        return r;
    }

    private static bool BreadthSearch(char[][] board, (int X, int Y) startPoint,
        Dictionary<(int X, int Y), (int X, int Y)[]> neighbours,
        string word)
    {
        var visited = new HashSet<(int, int)>();
        var queue = new Queue<(int, int)>();
        queue.Enqueue(startPoint);
        var pointer = 0;
        while (queue.Count != 0)
        {
            pointer++;
            if (pointer == word.Length)
                return true;
            var currentNode = queue.Dequeue();
            visited.Add(currentNode);
            foreach (var node in neighbours[currentNode])
            {
                if (!queue.Contains(node) &&
                    !visited.Contains(node) &&
                    board[node.Y][node.X] == word[pointer])
                {
                    queue.Enqueue(node);
                }
            }
        }

        return false;
    }

    public static string MakeGood(string line)
    {
        var sb = new StringBuilder(line.Length);
        var left = 0;
        var right = 1;
        var rejected = new HashSet<int>();
        var maxRight = -1;
        while (right < line.Length)
        {
            if (line[left] - 32 != line[right] && line[left] != line[right] - 32)
            {
                left++;
                right++;
                continue;
            }

            maxRight = Math.Max(maxRight, right);
            rejected.Add(left);
            rejected.Add(right);
            if (left == 0)
            {
                left = maxRight + 1;
                right = left + 1;
            }
            else
            {
                left--;
                right++;
            }
        }

        for (var i = 0; i < line.Length; i++)
        {
            if (rejected.Contains(i))
                continue;
            sb.Append(line[i]);
        }

        return sb.ToString();
    }

    public static string MinRemoveToMakeValid(string line)
    {
        var stack = new Stack<(char bracket, int position)>();
        var deleted = new HashSet<int>();
        for (var i = 0; i < line.Length; i++)
        {
            if (line[i] == '(')
                stack.Push(('(', i));
            else if (line[i] == ')')
            {
                if (stack.Count == 0)
                {
                    deleted.Add(i);
                }
                else
                {
                    stack.Pop();
                }
            }
        }

        while (stack.Count != 0)
            deleted.Add(stack.Pop().position);
        var sb = new StringBuilder(line.Length);
        for (var i = 0; i < line.Length; i++)
        {
            if (deleted.Contains(i))
                continue;
            sb.Append(line[i]);
        }

        return sb.ToString();
    }

    public static bool CheckValidString(string line)
    {
        var leftMin = 0;
        var leftMax = 0;
        foreach (var x in line)
        {
            if (x == '*')
            {
                leftMin = Math.Max(0, leftMin - 1);
                leftMax++;
            }
            else if (x == '(')
            {
                leftMin++;
                leftMax++;
            }
            else if (x == ')')
            {
                leftMin = Math.Max(0, leftMin - 1);
                leftMax--;
            }
        }

        return leftMin == 0;
    }

    public int CountStudents(int[] students, int[] sandwiches)
    {
        var queue = new Queue<int>();
        var sandwichPointer = 0;
        var preferCircle = 0;
        var preferSquare = 0;
        foreach (var x in students)
        {
            queue.Enqueue(x);
            if (x == 0)
                preferCircle++;
            else
                preferSquare++;
        }

        while (sandwichPointer < sandwiches.Length)
        {
            var student = queue.Dequeue();
            if (student == sandwiches[sandwichPointer])
            {
                sandwichPointer++;
                if (student == 0)
                    preferCircle--;
                else
                    preferSquare--;
            }
            else
            {
                queue.Enqueue(student);
                if (student == 0 && preferSquare == 0)
                    return queue.Count;
                if (student == 1 && preferCircle == 0)
                    return queue.Count;
            }
        }

        return 0;
    }

    public static int TimeRequiredToBuy(int[] tickets, int k)
    {
        var count = tickets[k];
        var time = 0;

        for (var i = 0; i < count; i++)
        {
            for (var j = 0; j < tickets.Length; j++)
            {
                if (tickets[j] == 0)
                    continue;
                if (tickets[k] == 0)
                    break;
                time++;
                tickets[j]--;
            }
        }

        // foreach (var x in tickets)
        // {
        //     if (x - count < 0)
        //     {
        //         time += x;
        //     }
        //     else
        //         time += count;
        // }

        return time;
    }

    public static string RemoveKdigits(string num, int k)
    {
        var result = new StringBuilder(num);
        var i = 0;
        if (num.Length == k)
            return "0";
        while (k > 0 && i < result.Length - 1)
        {
            if (result[i] > result[i + 1])
            {
                result.Remove(i, 1);
                k--;
            }
            else
            {
                result.Remove(i + 1, 1);
                i++;
            }
        }

        var x = 0;
        while (x < result.Length && result[x] == '0')
        {
            x++;
        }

        if (x == result.Length)
            return "0";
        return result.ToString().Substring(x);
    }

    public int[] RunningSum(int[] nums)
    {
        var sum = nums[0];
        for (var i = 1; i < nums.Length; i++)
        {
            sum += nums[i];
            nums[i] = sum;
        }

        return nums;
    }

    public static int Trap(int[] array)
    {
        var max = array.Max();
        var result = 0;
        for (var h = 0; h < max; h++)
        {
            var lastWall = -1;
            var slice = 0;
            var currentInSlice = 0;
            for (var i = 0; i < array.Length; i++)
            {
                if (array[i] - h > 0)
                {
                    lastWall = i;
                    slice += currentInSlice;
                    currentInSlice = 0;
                    continue;
                }

                if (lastWall >= 0)
                {
                    if (i == array.Length - 1)
                    {
                        currentInSlice = 0;
                    }

                    currentInSlice++;
                }
            }

            result += slice;
        }

        return result;
    }

    public int MaximalRectangle(char[][] matrix)
    {
        var height = matrix.Length;
        var width = matrix[0].Length;
        var leftUnits = new int[height][];
        for (var y = 0; y < height; y++)
        {
            leftUnits[y] = new int[width];
            for (var x = 0; x < width; x++)
            {
                if (x == 0)
                {
                    leftUnits[y][x] = 0;
                    continue;
                }

                if (matrix[y][x - 1] == '0')
                    continue;

                leftUnits[y][x] = leftUnits[y][x - 1] + 1;
            }
        }

        var maxArea = 0;
        for (var x = 0; x < width; x++)
        {
            var minWidth = int.MaxValue;
            for (var y = 0; y < height; y++)
            {
                if (leftUnits[y][x] == 0)
                    continue;
                minWidth = Math.Min(minWidth, leftUnits[y][x]);
                maxArea = Math.Max(minWidth * (y + 1), maxArea);
            }
        }

        return maxArea;
    }

    public int MostWordsFound(string[] sentences)
    {
        var maxWhitespaces = 0;
        foreach (var line in sentences)
        {
            var whitespaces = 0;
            foreach (var x in line)
            {
                if (x != ' ')
                    continue;

                whitespaces++;
                maxWhitespaces = Math.Max(whitespaces + 1, maxWhitespaces);
            }
        }

        return maxWhitespaces;
    }

    public static string FinalString(string s)
    {
        var left = -1;
        var right = 0;
        var dictionary = new Dictionary<int, char>();
        var reverse = false;
        var start = 0;
        while (s[start] == 'i')
            start++;

        for (var i = start; i < s.Length; i++)
        {
            if (s[i] == 'i')
                reverse = !reverse;
            else
                dictionary.Add(reverse ? left-- : right++, s[i]);
        }

        if (left == -1)
            return s;

        var chars = new char[s.Length];
        // var chars = Enumerable.Range(0, s.Length)
        // .Select(i => dictionary[i - left])
        // .ToArray();

        if (!reverse)
            for (var i = 0; i < right - left - 1; i++)
            {
                chars[i] = dictionary[i - left];
            }
        else
            for (var i = right - left - 2; i >= 0; i--)
            {
                chars[i] = dictionary[i + left];
            }


        return new string(chars);
    }

    public int IslandPerimeter(int[][] grid)
    {
        var height = grid.Length;
        var width = grid[0].Length;
        var perimeter = 0;
        for (var y = 0; y < height; y++)
        for (var x = 0; x < width; x++)
        {
            if (grid[y][x] == '0')
                continue;
            if (y - 1 < 0 || grid[y - 1][x] == '0')
                perimeter++;
            if (y + 1 == height || grid[y + 1][x] == '0')
                perimeter++;
            if (x - 1 < 0 || grid[y][x - 1] == '0')
                perimeter++;
            if (x + 1 == width || grid[y][x + 1] == '0')
                perimeter++;
        }

        return perimeter;
    }

    public static int NumIslands(char[][] grid)
    {
        var height = grid.Length;
        var width = grid.Length > 0 ? grid[0].Length : 0;
        var neighbours = Enumerable.Range(0, height)
            .SelectMany(y => Enumerable.Range(0, width).Select(x => (x, y)))
            .Where(p => grid[p.y][p.x] == '1')
            .Select(p => (p, new List<(int x, int y)>
            {
                (p.x - 1, p.y),
                (p.x + 1, p.y),
                (p.x, p.y - 1),
                (p.x, p.y + 1)
            }.Where(i => i.x >= 0 && i.x < width &&
                         i.y >= 0 && i.y < height &&
                         grid[i.y][i.x] == '1').ToArray()))
            .ToDictionary(i => i.p, i => i.Item2);
        var visited = new HashSet<(int x, int y)>();
        var result = 0;
        foreach (var point in neighbours.Keys)
        {
            if (visited.Contains(point))
                continue;
            result++;
            var island = Bfs(point, neighbours);
            visited.Add(point);
            foreach (var p in island)
            {
                visited.Add(p);
            }
        }

        return result;
    }

    private static HashSet<(int x, int y)> Bfs((int X, int Y) startPoint,
        Dictionary<(int X, int Y), (int X, int Y)[]> neighbours)
    {
        var visited = new HashSet<(int, int)>
        {
            startPoint
        };
        var queue = new Queue<(int, int)>();
        queue.Enqueue(startPoint);
        while (queue.Count != 0)
        {
            var currentNode = queue.Dequeue();
            foreach (var node in neighbours[currentNode])
            {
                if (!queue.Contains(node) && !visited.Contains(node))
                {
                    queue.Enqueue(node);
                    visited.Add(node);
                }
            }
        }

        return visited;
    }

    void MakeDictionary()
    {
        var weights = Enumerable.Range(0, int.Parse(Console.ReadLine()))
            .Select(_ => Console.ReadLine().Split(' '))
            .ToDictionary(x => x[0], x => int.Parse(x[1]));
        var inputs = Enumerable.Range(0, int.Parse(Console.ReadLine()))
            .Select(_ => Console.ReadLine());
        var indexedDict = new Dictionary<string, List<string>>();
        foreach (var word in weights.Keys)
        {
            var sb = new StringBuilder(word.Length);
            foreach (var x in word)
            {
                sb.Append(x);
                var part = sb.ToString();
                if (!indexedDict.ContainsKey(part))
                    indexedDict.Add(part, new List<string>());
                indexedDict[part].Add(word);
            }
        }

        var result = new List<string>();
        foreach (var input in inputs)
        {
            if (!indexedDict.ContainsKey(input))
                continue;
            var autocompletions = indexedDict[input]
                .OrderBy(x => weights[x])
                .ThenBy(x => x)
                .Take(10)
                .ToArray();
            result.Add('\n' + input + string.Join('\n', autocompletions));
        }
    }

    public static int LongestIdealString(string line, int k)
    {
        var maxLength = 0;
        var visited = new HashSet<int>();
        for (var i = 0; i < line.Length - 1; i++)
        {
            var length = 0;
            // if (visited.Contains(i))
            //     continue;
            var start = i;
            for (var j = i + 1; j < line.Length; j++)
            {
                if (Math.Abs(line[start] - line[j]) > k)
                    continue;
                if (!visited.Contains(start))
                    length++;
                length++;
                visited.Add(start);
                visited.Add(j);
                start = j;
            }

            maxLength = Math.Max(length, maxLength);
        }

        return maxLength;
    }

    public int[] SeparateDigits(int[] nums)
    {
        var list = new List<int>();
        foreach (var x in nums)
        {
            list.AddRange(Separate(x).Reverse());
        }

        return list.ToArray();
    }

    IEnumerable<int> Separate(int x)
    {
        while (x > 0)
        {
            yield return x % 10;
            x /= 10;
        }
    }

    public int TriangularSum(int[] nums)
    {
        var result = new List<int>(nums);
        while (result.Count > 1)
        {
            var buffer = new List<int>();
            for (var i = 0; i < result.Count - 1; i++)
            {
                buffer.Add(result[i] + result[i + 1]);
            }

            result = new List<int>(buffer);
        }

        return result.First();
    }

    public static string FreqAlphabets(string line)
    {
        var alphabet = Enumerable.Range(0, 26)
            .Select(x => (char)('a' + x))
            .ToArray();
        var sb = new List<char>();
        for (var i = line.Length - 1; i >= 0; i--)
        {
            if (line[i] == '#')
            {
                sb.Add(alphabet[10 * (line[i - 2] - '0') + line[i - 1] - '0' - 1]);
                i -= 2;
                continue;
            }

            sb.Add(alphabet[line[i] - '0' - 1]);
        }

        sb.Reverse();
        return new string(sb.ToArray());
    }

    public string DestCity(IList<IList<string>> paths)
    {
        var source = new HashSet<string>();
        foreach (var path in paths)
        {
            source.Add(path[0]);
        }

        foreach (var path in paths)
            if (!source.Contains(path[1]))
                return path[1];
        return string.Empty;
    }

    public string MergeAlternately(string word1, string word2)
    {
        var pointer = 0;
        var result = new char[word1.Length + word2.Length];
        var a = 0;
        var b = 0;
        while (pointer < result.Length - 1)
        {
            if (a < word1.Length)
                result[pointer++] = word1[a++];
            if (b < word2.Length)
                result[pointer++] = word2[b++];
        }

        return new string(result);
    }

    public static bool ContainsNearbyDuplicate(int[] nums, int k)
    {
        var dict = new Dictionary<int, HashSet<int>>();
        for (var i = 0; i < nums.Length; i++)
        {
            if (!dict.ContainsKey(nums[i]))
                dict.Add(nums[i], new HashSet<int>());
            dict[nums[i]].Add(i);
            if (dict[nums[i]].Count == 2)
            {
                var min = dict[nums[i]].Min();
                dict[nums[i]].Remove(min);
            }
        }

        return false;
    }

    public bool WordPattern(string pattern, string s)
    {
        var words = s.Split(' ');
        if (words.Length != pattern.Length)
            return false;
        var dict = new Dictionary<char, string>();
        var visited = new Dictionary<string, char>();
        for (var i = 0; i < words.Length; i++)
        {
            if (!dict.ContainsKey(pattern[i]))
            {
                if (!visited.ContainsKey(words[i]))
                    visited.Add(words[i], pattern[i]);
                else
                {
                    return false;
                }
                dict.Add(pattern[i], words[i]);
            }
            else
            {
                if (words[i] != dict[pattern[i]])
                    return false;
            }
        }

        return true;

        return true;
    }
    
}