using System.Diagnostics.Metrics;
using System.Text;
using System.Xml.Linq;
using Newtonsoft.Json;
using QuickType;

namespace CodeSandbox.Leetcode.Medium;

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

        static void Main(string[] args)
        {
            
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

        public static int MaxScore(string s)
        {
            var maxScore = s.Count(x => x == '1');

            if (s[0] == '0')
                maxScore++;
            else
                maxScore--;
            if (s[s.Length - 1] == '0')
                maxScore--;
            var score = maxScore;
            for (var i = 1; i < s.Length - 1; i++)
            {
                if (s[i] == '0')
                {
                    score++;
                }
                else
                {
                    score--;
                }

                maxScore = Math.Max(score, maxScore);
            }

            return maxScore;
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

        public static int Gcd(int a, int b)
        {
            if (b == 0)
                return a;
            return Gcd(b, a % b);
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

        public int MinOperations(string[] logs)
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

        public bool IsArraySpecial(int[] nums)
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

        public int[] MinOperations(string boxes)
        {
            var result = new int[boxes.Length];
            for (var i = 0; i < boxes.Length; i++)
            for (var j = 0; j < boxes.Length; j++)
            {
                if (i == j)
                    continue;
                result[i] += Math.Abs(i - j) * j;
            }

            return result;
        }

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