using System.Text;
using CodeSandbox.Leetcode.Medium;
using Console = System.Console;

namespace CodeSandbox;

public class Solution
{
    long[,] dp = new long[15, 10001];
    long[,] pr = new long[15, 10001];
    long[] tot = new long[15];
    const long mod = 1000000007;
    int n, mx;

    void Get(int la, int cn)
    {
        tot[cn]++;
        for (int p = 2 * la; p <= mx; p += la)
            Get(p, cn + 1);
    }

    public int IdealArrays(int nn, int mmx)
    {
        n = nn;
        mx = mmx;

        for (int i = 1; i <= 10000; i++)
        {
            dp[1, i] = 1;
            pr[1, i] = i;
        }

        for (int i = 2; i < 15; i++)
        {
            for (int j = i; j <= 10000; j++)
            {
                dp[i, j] = pr[i - 1, j - 1];
                pr[i, j] = (dp[i, j] + pr[i, j - 1]) % mod;
            }
        }

        for (int i = 1; i <= mx; i++)
            Get(i, 1);

        long ans = mx;
        for (int i = 2; i < 15; i++)
        {
            ans = (ans + tot[i] * dp[i, n]) % mod;
        }

        return (int)ans;
    }
}

public class Meow1
{
    public class FenwickTree
    {
        private int[] tree;

        public FenwickTree(int size)
        {
            tree = new int[size + 1];
        }

        public void Update(int index, int delta)
        {
            index++;
            while (index < tree.Length)
            {
                tree[index] += delta;
                index += index & -index;
            }
        }

        public int Query(int index)
        {
            index++;
            int res = 0;
            while (index > 0)
            {
                res += tree[index];
                index -= index & -index;
            }

            return res;
        }

        public int MaxTaskAssign1(int[] tasks, int[] workers, int pills, int strength)
        {
            Array.Sort(tasks, (a, b) => b.CompareTo(a));
            Array.Sort(workers, (a, b) => b.CompareTo(a));

            var taskIndex = 0;
            var workerIndex = 0;
            var pillWorkerPointer = 0;
            var completedTasks = 0;
            var scanningDown = true;
            var pillQueue = new PriorityQueue<int, int>(Comparer<int>.Create((a, b) => b.CompareTo(a)));

            while (taskIndex < tasks.Length)
            {
                while (workerIndex < workers.Length && workers[workerIndex] == -1)
                    workerIndex++;
                if (workerIndex >= workers.Length)
                {
                    // No more normal workers available, try pill-assisted ones
                    if (pillQueue.Count > 0 && pillQueue.Peek() >= tasks[taskIndex])
                    {
                        pillQueue.Dequeue();
                        completedTasks++;
                    }
                }
                else if (workers[workerIndex] >= tasks[taskIndex])
                {
                    // Worker can do the task without a pill
                    workers[workerIndex] = -1;
                    workerIndex++;
                    completedTasks++;
                }
                else
                {
                    if (scanningDown)
                    {
                        while (pillWorkerPointer < workers.Length &&
                               (workers[pillWorkerPointer] == -1 ||
                                workers[pillWorkerPointer] + strength >= tasks[taskIndex]))
                        {
                            pillWorkerPointer++;
                        }

                        if (pillWorkerPointer == workers.Length)
                        {
                            scanningDown = false;
                            pillWorkerPointer = workers.Length - 1;
                        }
                    }

                    while (pillWorkerPointer >= workers.Length ||
                           (pillWorkerPointer > workerIndex &&
                            (workers[pillWorkerPointer] == -1 ||
                             workers[pillWorkerPointer] + strength < tasks[taskIndex])))
                    {
                        pillWorkerPointer--;
                    }

                    if (pillQueue.Count >= pills && pillQueue.Count > 0 && pillQueue.Peek() >= tasks[taskIndex])
                    {
                        pillQueue.Dequeue();
                        completedTasks++;
                    }
                    else if (pillWorkerPointer >= 0 && workers[pillWorkerPointer] + strength >= tasks[taskIndex])
                    {
                        pillQueue.Enqueue(workers[pillWorkerPointer], workers[pillWorkerPointer]);
                        workers[pillWorkerPointer] = -1;
                        pillWorkerPointer = scanningDown ? pillWorkerPointer - 1 : pillWorkerPointer + 1;
                    }
                    else if (pillQueue.Count > 0 && pillQueue.Peek() >= tasks[taskIndex])
                    {
                        pillQueue.Dequeue();
                        completedTasks++;
                    }
                }

                taskIndex++;
                pillWorkerPointer = Math.Max(pillWorkerPointer, workerIndex);
            }

            return completedTasks + Math.Min(pillQueue.Count, pills);
        }

        static void Main()
        {
        }
        

        public int MinimumScore(int[] nums, int[][] edges)
        {
            var n = nums.Length;

            var tree = new List<int>[n];
            for (var i = 0; i < n; i++)
                tree[i] = new List<int>();

            foreach (var edge in edges)
            {
                int a = edge[0], b = edge[1];
                tree[a].Add(b);
                tree[b].Add(a);
            }

            var xor = new int[n];
            var parent = new int[n];
            Array.Fill(parent, -1);

            DFS(0, -1, nums, tree, xor, parent);

            var totalXor = xor[0];
            var minScore = int.MaxValue;

            for (var u = 1; u < n; u++)
            {
                for (var v = u + 1; v < n; v++)
                {
                    int a = xor[u], b = xor[v], c;

                    if (IsAncestor(u, v, parent))
                    {
                        c = totalXor ^ a;
                        a ^= b;
                    }
                    else if (IsAncestor(v, u, parent))
                    {
                        c = totalXor ^ b;
                        b ^= a;
                    }
                    else
                    {
                        c = totalXor ^ a ^ b;
                    }

                    int max = Math.Max(a, Math.Max(b, c));
                    int min = Math.Min(a, Math.Min(b, c));
                    minScore = Math.Min(minScore, max - min);
                }
            }

            return minScore;
        }

        private int DFS(int node, int par, int[] nums, List<int>[] tree, int[] xor, int[] parent)
        {
            xor[node] = nums[node];
            parent[node] = par;

            foreach (var nei in tree[node])
            {
                if (nei != par)
                {
                    xor[node] ^= DFS(nei, node, nums, tree, xor, parent);
                }
            }

            return xor[node];
        }

        private bool IsAncestor(int u, int v, int[] parent)
        {
            while (v != -1)
            {
                if (v == u) 
                    return true;
                v = parent[v];
            }

            return false;
        }

        public static int MaximumGain(string s, int x, int y)
        {
            var maxGain = 0;
            var gain = 0;
            var stack = new Stack<char>(s.Length);
            var first = 'b';
            var second = 'a';
            if (x > y)
            {
                (x, y) = (y, x);
                (first, second) = (second, first);
            }

            foreach (var c in s)
            {
                if (c != second)
                {
                    stack.Push(c);
                    continue;
                }

                if (stack.Count > 0 && stack.Peek() == first)
                {
                    stack.Pop();
                    gain += y;
                    maxGain = Math.Max(maxGain, gain);
                }
                else
                    stack.Push(c);
            }

            var reversedStack = new Stack<char>(stack.Count);
            while (stack.Count > 0)
            {
                var c = stack.Pop();
                if (c != second)
                {
                    reversedStack.Push(c);
                    continue;
                }

                if (reversedStack.Count > 0 && reversedStack.Peek() == first)
                {
                    reversedStack.Pop();
                    gain += x;
                    maxGain = Math.Max(maxGain, gain);
                }
                else
                    reversedStack.Push(c);
            }

            return maxGain;
        }

        public static int MaximumUniqueSubarray(int[] nums)
        {
            var buffer = new HashSet<int>(nums.Length);
            var sum = 0;
            var max = 0;
            var left = 0;
            for (var i = 0; i < nums.Length; i++)
            {
                if (buffer.Contains(nums[i]))
                {
                    while (left < i && buffer.Contains(nums[i]))
                    {
                        sum -= nums[left];
                        buffer.Remove(nums[left]);
                        left++;
                    }
                }

                sum += nums[i];
                max = Math.Max(max, sum);
                buffer.Add(nums[i]);
            }

            return max;
        }

        public static string MakeFancyString(string s)
        {
            var count = 1;
            var sb = new StringBuilder(s.Length);
            sb.Append(s[0]);
            for (var i = 1; i < s.Length; i++)
            {
                if (s[i] == s[i - 1])
                    count++;
                else
                    count = 1;
                if (count < 3)
                    sb.Append(s[i]);
            }

            return sb.ToString();
        }

        class Node
        {
            public string Name { get; set; }
            public Dictionary<string, Node> Children { get; set; }

            public static Node Create(string name)
            {
                return new Node
                {
                    Name = name,
                    Children = new Dictionary<string, Node>()
                };
            }

            public static void Add(Node root, IList<string> path)
            {
                var node = root;
                foreach (var child in path)
                {
                    node.Children.TryAdd(child, Create(child));
                    node = node.Children[child];
                }
            }
        }

        public IList<IList<string>> DeleteDuplicateFolder(IList<IList<string>> paths)
        {
            var root = Node.Create("/");
            foreach (var path in paths)
                Node.Add(root, path);
            var subtreeCount = new Dictionary<string, int>();
            var nodeToSerialization = new Dictionary<Node, string>();
            SerializeSubtree(root, subtreeCount, nodeToSerialization);
            var remainingPaths = new List<IList<string>>();
            CollectPaths(root, new List<string>(), remainingPaths, subtreeCount, nodeToSerialization);
            return remainingPaths;
        }

        private string SerializeSubtree(
            Node node,
            Dictionary<string, int> subtreeCount,
            Dictionary<Node, string> nodeToSerialization)
        {
            if (node.Children.Count == 0)
                return string.Empty;

            var serializedChildren = node.Children
                .Values
                .Select(child => child.Name + "(" + SerializeSubtree(child, subtreeCount, nodeToSerialization) + ")")
                .ToList();

            serializedChildren.Sort();
            var serialized = string.Join(",", serializedChildren);

            if (!subtreeCount.TryAdd(serialized, 1))
                subtreeCount[serialized]++;

            nodeToSerialization[node] = serialized;
            return serialized;
        }

        private void CollectPaths(Node node,
            List<string> currentPath,
            IList<IList<string>> remainingPaths,
            Dictionary<string, int> subtreeCount,
            Dictionary<Node, string> nodeToSerialization)
        {
            if (node == null)
                return;

            if (nodeToSerialization.ContainsKey(node))
            {
                var serialized = nodeToSerialization[node];
                if (subtreeCount.ContainsKey(serialized) && subtreeCount[serialized] > 1)
                    return;
            }

            if (!node.Name.Equals("/"))
                remainingPaths.Add(new List<string>(currentPath));
            foreach (var child in node.Children)
            {
                currentPath.Add(child.Key);
                CollectPaths(child.Value, currentPath, remainingPaths, subtreeCount, nodeToSerialization);
                currentPath.RemoveAt(currentPath.Count - 1);
            }
        }

        public IList<string> RemoveSubfolders(string[] folders)
        {
            var deleted = new HashSet<int>(folders.Length);
            Array.Sort(folders);
            for (var i = 0; i < folders.Length; i++)
            {
                if (deleted.Contains(i))
                    continue;
                for (var j = i + 1; j < folders.Length; j++)
                {
                    if (deleted.Contains(j))
                        continue;
                    if (Contains(folders[i], folders[j]))
                        deleted.Add(j);
                }
            }

            var heads = new List<string>(folders.Length - deleted.Count);

            for (var i = 0; i < folders.Length; i++)
                if (!deleted.Contains(i))
                    heads.Add(folders[i]);

            return heads;
        }

        bool Contains(string folder, string subFolder)
        {
            if (subFolder.Length < folder.Length)
                return false;
            for (var i = 0; i < folder.Length; i++)
                if (subFolder[i] != folder[i])
                    return false;
            return subFolder[folder.Length] == '/';
        }

        public long MinimumDifference(int[] nums)
        {
            var n = nums.Length / 3;
            var sum = 0L;

            var maxHeap = new PriorityQueue<long, long>(Comparer<long>.Create((a, b) => b.CompareTo(a)));
            var minHeap = new PriorityQueue<long, long>();
            var prefix = new long[nums.Length];
            var suffix = new long[nums.Length];
            for (var i = 0; i < nums.Length; i++)
            {
                sum += nums[i];
                maxHeap.Enqueue(nums[i], nums[i]);
                if (maxHeap.Count > n)
                    sum -= maxHeap.Dequeue();
                if (maxHeap.Count >= n)
                    prefix[i] = sum;
            }

            sum = 0;
            for (var i = nums.Length - 1; i >= 0; i--)
            {
                sum += nums[i];
                minHeap.Enqueue(nums[i], nums[i]);
                if (minHeap.Count > n)
                    sum -= minHeap.Dequeue();
                if (minHeap.Count >= n)
                    suffix[i] = sum;
            }

            var ans = long.MaxValue;

            for (var i = n - 1; i < nums.Length - n; i++)
                ans = Math.Min(ans, prefix[i] - suffix[i + 1]);

            return ans;
        }

        public int MaximumLength(int[] nums, int k)
        {
            var max = 0;
            var dp = new int[k, k];
            foreach (var x in nums)
            {
                var remainder = x % k;
                for (var i = 0; i < k; i++)
                {
                    dp[i, remainder] = dp[remainder, i] + 1;
                    max = Math.Max(max, dp[i, remainder]);
                }
            }

            return max;
        }

        public int MaximumLength(int[] nums)
        {
            var x = Meow(nums, 0, 0);
            var y = Meow(nums, 1, 1);
            var u = Meow(nums, 1, 0);
            var w = Meow(nums, 0, 1);
            return Math.Max(Math.Max(x, y), Math.Max(u, w));
        }

        int Meow(int[] nums, int startRemainder, int remainder)
        {
            var start = 0;
            while (start < nums.Length && nums[start] % 2 != startRemainder)
                start++;
            if (start == nums.Length)
                return 0;
            var prev = nums[start];
            var count = 1;
            for (var i = start + 1; i < nums.Length; i++)
            {
                if ((prev + nums[i]) % 2 != remainder)
                    continue;
                count++;
                prev = nums[i];
            }

            return count;
        }

        public bool IsValid(string word)
        {
            if (word.Length < 3)
                return false;
            var hasVowel = false;
            var hasConsolant = false;
            var vowels = new[] { 'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U' };
            foreach (var x in word)
            {
                if (vowels.Contains(x))
                    hasVowel = true;
                else if (char.IsLetter(x))
                    hasConsolant = true;
                else if (!char.IsDigit(x))
                    continue;
                else
                    return false;
            }

            Console.WriteLine(hasVowel);
            Console.WriteLine(hasConsolant);
            return hasVowel && hasConsolant;
        }

        public int GetDecimalValue(RangeBitwiseAnd1.Solution.ListNode head)
        {
            var current = 0;
            while (head is not null)
            {
                current += head.val;
                current <<= 1;
                head = head.next;
            }

            return current;
        }

        public static int MatchPlayersAndTrainers(int[] players, int[] trainers)
        {
            Array.Sort(players);
            Array.Sort(trainers);
            var p = players.Length - 1;
            var t = trainers.Length - 1;
            var matches = 0;
            while (p >= 0 && t >= 0)
            {
                while (p >= 0 && trainers[t] < players[p])
                    p--;
                if (p < 0)
                    break;
                matches++;
                t--;
                p--;
            }

            return matches;
        }

        public int[] EarliestAndLatest(int n, int firstPlayer, int secondPlayer)
        {
            var left = Math.Min(firstPlayer, secondPlayer);
            var right = Math.Max(firstPlayer, secondPlayer);
            if (left + right == n + 1)
                return new[] { 1, 1 };
            if (n == 3 || n == 4)
                return new[] { 2, 2 };
            if (left - 1 > n - right)
            {
                var temp = n + 1 - left;
                left = n + 1 - right;
                right = temp;
            }

            var nextRound = (n + 1) / 2;
            var minRound = n;
            var maxRound = 1;

            if (right * 2 <= n + 1)
            {
                var preLeft = left - 1;
                var midGap = right - left - 1;
                for (var i = 0; i <= preLeft; ++i)
                for (var j = 0; j <= midGap; ++j)
                {
                    var res = EarliestAndLatest(nextRound, i + 1, i + j + 2);
                    minRound = Math.Min(minRound, 1 + res[0]);
                    maxRound = Math.Max(maxRound, 1 + res[1]);
                }
            }
            else
            {
                var mirrored = n + 1 - right;
                var preLeft = left - 1;
                var midGap = mirrored - left - 1;
                var innerGap = right - mirrored - 1;

                for (var i = 0; i <= preLeft; ++i)
                for (var j = 0; j <= midGap; ++j)
                {
                    var pos1 = i + 1;
                    var pos2 = i + j + 1 + (innerGap + 1) / 2 + 1;
                    var res = EarliestAndLatest(nextRound, pos1, pos2);
                    minRound = Math.Min(minRound, 1 + res[0]);
                    maxRound = Math.Max(maxRound, 1 + res[1]);
                }
            }

            return new[] { minRound, maxRound };
        }

        public static int MostBooked(int n, int[][] meetings)
        {
            var rooms = new PriorityQueue<int, int>();
            var endTimes = new PriorityQueue<(long End, int Room), (long, int)>();
            var times = meetings
                .Select(x => new { Start = x.First(), End = x.Last() })
                .OrderBy(x => x.Start)
                .ToArray();
            var bookings = new int[n];
            var max = 1;
            var highloadRoom = 0;
            for (var i = 0; i < n; i++)
                rooms.Enqueue(i, i);
            foreach (var t in times)
            {
                var room = -1;
                var end = -1L;
                while (endTimes.Count > 0 && endTimes.Peek().End <= t.Start)
                {
                    (end, room) = endTimes.Dequeue();
                    rooms.Enqueue(room, room);
                }

                if (rooms.Count > 0)
                    room = rooms.Dequeue();
                if (room == -1)
                    (end, room) = endTimes.Dequeue();
                bookings[room]++;
                if (bookings[room] > max || (bookings[room] == max && room < highloadRoom))
                {
                    max = bookings[room];
                    highloadRoom = room;
                }

                var newEnd = t.End + Math.Max(0, end - t.Start);
                endTimes.Enqueue((newEnd, room), (newEnd, room));
            }

            return highloadRoom;
        }

        public static int[] SumZero(int n)
        {
            var result = new int[n];
            var current = 1;
            for (var i = 0; i < n / 2 + n % 2; i++)
            {
                if (n % 2 == 1 && i == 0)
                    continue;
                result[i + n / 2] = current;
                result[n / 2 - i - 1 + n % 2] = -current;
                current++;
            }

            return result;
        }

        public int MaxFreeTime(int eventTime, int k, int[] startTime, int[] endTime)
        {
            var gaps = 0;
            var meow = new int[startTime.Length + 1];
            for (var i = 0; i < startTime.Length; i++)
                meow[i + 1] = meow[i] + endTime[i] - startTime[i];
            for (var i = k - 1; i < startTime.Length; i++)
            {
                var right = i == startTime.Length - 1 ? eventTime : startTime[i + 1];
                var left = i == k - 1 ? 0 : endTime[i - k];
                gaps = Math.Max(right - left - meow[i + 1] + meow[i + 1 - k], gaps);
            }

            return gaps;
        }

        public int MaxFreeTime(int eventTime, int[] startTime, int[] endTime)
        {
            var gaps = 0;
            var previousGap = startTime[0];
            var nextGap = 0;
            var maxGap = 0;
            for (var i = 0; i < startTime.Length; i++)
            {
                nextGap = i == startTime.Length - 1
                    ? eventTime - endTime[i]
                    : startTime[i + 1] - endTime[i];

                var duration = endTime[i] - startTime[i];
                var idealFreeTime = previousGap + nextGap + duration;
                gaps = duration <= maxGap
                    ? Math.Max(gaps, idealFreeTime)
                    : Math.Max(gaps, previousGap + nextGap);

                maxGap = Math.Max(maxGap, previousGap);
                previousGap = nextGap;
            }

            previousGap = eventTime - endTime[startTime.Length - 1];
            maxGap = 0;
            for (var id = startTime.Length - 1; id >= 0; id--)
            {
                nextGap = id == 0
                    ? startTime[id]
                    : startTime[id] - endTime[id - 1];

                var duration = endTime[id] - startTime[id];
                var idealFreeTime = previousGap + nextGap + duration;
                if (duration <= maxGap)
                    gaps = Math.Max(gaps, idealFreeTime);

                maxGap = Math.Max(maxGap, previousGap);
                previousGap = nextGap;
            }

            return gaps;
        }

        public int MaxValue(int[][] events, int k)
        {
            Array.Sort(events, (a, b) => a[0] - b[0]);
            var n = events.Length;

            var dp = new int[k + 1][];
            for (var i = 0; i <= k; i++)
            {
                dp[i] = new int[n];
                Array.Fill(dp[i], -1);
            }

            return DepthSearch(0, k, events, dp);
        }

        private int DepthSearch(int curIndex, int count, int[][] events, int[][] dp)
        {
            if (count == 0 || curIndex == events.Length)
                return 0;

            if (dp[count][curIndex] != -1)
                return dp[count][curIndex];

            var nextIndex = BisectRight(events, events[curIndex][1]);
            dp[count][curIndex] = Math.Max(DepthSearch(curIndex + 1, count, events, dp),
                events[curIndex][2] + DepthSearch(nextIndex, count - 1, events, dp));
            return dp[count][curIndex];
        }

        public static int BisectRight(int[][] events, int target)
        {
            var left = 0;
            var right = events.Length;
            while (left < right)
            {
                var mid = (left + right) / 2;
                if (events[mid][0] <= target)
                    left = mid + 1;
                else
                    right = mid;
            }

            return left;
        }

        public int MaxEvents(int[][] events)
        {
            Array.Sort(events, (a, b) => a[1] == b[1] ? a[0] - b[0] : a[1] - b[1]);

            var seen = new bool[100001];
            var count = 0;
            var last = 0;
            for (var i = 0; i < events.Length; i++)
            {
                var end = events[i][1];
                var start = i > 0 && events[i - 1][0] == events[i][0]
                    ? last
                    : events[i][0];

                for (var j = start; j <= end; j++)
                {
                    if (seen[j])
                        continue;
                    seen[j] = true;
                    last = j;
                    count++;
                    break;
                }
            }

            return count;
        }

        public class FindSumPairs
        {
            private Dictionary<int, int> first;
            private Dictionary<int, int> second;
            private int[] nums2;

            public FindSumPairs(int[] nums1, int[] nums2)
            {
                first = new Dictionary<int, int>(1000);
                second = new Dictionary<int, int>();
                foreach (var x in nums1)
                {
                    first.TryAdd(x, 0);
                    first[x]++;
                }

                foreach (var x in nums2)
                {
                    second.TryAdd(x, 0);
                    second[x]++;
                }

                this.nums2 = nums2;
            }

            public void Add(int index, int val)
            {
                var previous = nums2[index];
                nums2[index] += val;
                var value = nums2[index];
                second.TryAdd(value, 0);
                second[value]++;
                second[previous]--;
                if (second[previous] == 0)
                    second.Remove(previous);
            }

            public int Count(int tot)
            {
                var count = 0;
                foreach (var x in first.Keys)
                {
                    if (second.ContainsKey(tot - x))
                        count += second[x] * first[x];
                }

                return count;
            }
        }

        public int FindLucky(int[] arr)
        {
            var frequency = new Dictionary<int, int>(arr.Length);
            foreach (var x in arr)
            {
                frequency.TryAdd(x, 0);
                frequency[x]++;
            }

            var lucky = -1;
            foreach (var f in frequency)
            {
                if (f.Value != f.Key)
                    continue;
                lucky = Math.Max(lucky, f.Key);
            }

            return lucky;
        }

        public static char KthCharacter(long k, int[] operations)
        {
            var operation = 0;
            var length = 1L;
            for (var i = 0; i < operations.Length; i++)
            {
                length <<= 1;
                if (length < k)
                    continue;
                operation = i;
                break;
            }

            for (var i = Math.Min(operation + 1, 26); i >= 0; i--)
            {
                var position = k - 1;
                var currentLength = length;
                var symbol = (char)('a' + i);
                for (var o = operation; o >= 0; o--)
                {
                    var currentOperation = operations[o];
                    if (position >= currentLength / 2 && currentOperation == 1)
                        symbol = (char)('a' + (26 + symbol - 'a' - 1) % 26);

                    position %= currentLength / 2;
                    currentLength /= 2;
                }

                if (symbol == 'a')
                    return (char)('a' + i);
            }

            return '\0';
        }

        public int PossibleStringCount(string word, int k)
        {
            var equal = 1;
            var result = 1L;
            var frequency = new List<int>(word.Length);
            for (var i = 1; i < word.Length; i++)
            {
                if (word[i] == word[i - 1])
                    equal++;
                else
                {
                    frequency.Add(equal);
                    equal = 1;
                }
            }

            frequency.Add(equal);

            var module = 1_000_000_007;
            foreach (var x in frequency)
                result = result * x % module;

            if (frequency.Count > k)
                return (int)result;

            var f = new int[k];
            var prefixSum = new int[k];
            f[0] = 1;
            Array.Fill(prefixSum, 1);
            foreach (var x in frequency)
            {
                var newFrequency = new int[k];
                for (var j = 1; j < k; ++j)
                {
                    newFrequency[j] = prefixSum[j - 1];
                    if (j - x - 1 < 0)
                        continue;
                    newFrequency[j] = (newFrequency[j] - prefixSum[j - x - 1] + module) % module;
                }

                var newPrefixSum = new int[k];
                newPrefixSum[0] = newFrequency[0];
                for (var j = 1; j < k; ++j)
                    newPrefixSum[j] = (newPrefixSum[j - 1] + newFrequency[j]) % module;

                f = newFrequency;
                prefixSum = newPrefixSum;
            }

            return (int)((result - prefixSum[k - 1] + module) % module);
        }

        public static int PossibleStringCount(string word)
        {
            var equal = 1;
            var sum = 0;
            var subsequencesCount = 0;
            for (var i = 1; i < word.Length; i++)
            {
                if (word[i] == word[i - 1])
                    equal++;
                else
                {
                    sum += equal;
                    subsequencesCount++;
                    equal = 1;
                }
            }

            sum += equal;
            subsequencesCount++;
            return 1 + sum - subsequencesCount;
        }

        public static int FindLHS(int[] nums)
        {
            Array.Sort(nums);
            var frequency = new Dictionary<int, int>(nums.Length);
            foreach (var x in nums)
            {
                frequency.TryAdd(x, 0);
                frequency[x]++;
            }

            var keys = frequency.Keys.ToArray();
            Array.Sort(keys);
            var max = 0;
            for (var i = 0; i < keys.Length - 1; i++)
            {
                if (keys[i] - keys[i - 1] != 1)
                    continue;
                max = Math.Max(max, frequency[keys[i]] + frequency[keys[i - 1]]);
            }

            return max;
        }

        public int[] MaxSubsequence(int[] nums, int k)
        {
            return nums.Select((x, i) => (x, i))
                .OrderBy(x => x.x)
                .ThenBy(x => x.i)
                .Skip(nums.Length - k)
                .Take(k)
                .OrderBy(x => x.i)
                .Select(x => x.x)
                .ToArray();
        }

        public static int NumSubseq(int[] nums, int target)
        {
            Array.Sort(nums);
            var maxRight = 0;
            var array = new int[nums.Length];
            array[0] = 1;
            var module = 1000000007;
            for (var i = 0; i < nums.Length; i++)
            {
                array[i] = 2 * nums[i - 1] % module;
            }

            var right = nums.Length - 1;
            var left = 0;
            while (left <= right)
            {
                if (nums[left] + nums[right] <= target)
                {
                    maxRight = (maxRight + array[right - left]) % module;
                    left++;
                }
                else
                    right--;
            }

            // for (var left = 0; left < nums.Length; left++)
            // {
            //     right = Math.Max(right, left);
            //     while (right + 1 < nums.Length && nums[left] + nums[right + 1] <= target)
            //     {
            //         right++;
            //     }
            //
            //     array[left] = right;
            //     maxRight = Math.Max(right, maxRight);
            // }

            return maxRight;
        }

        public static string LongestSubsequenceRepeatedK(string s, int k)
        {
            var frequency = new int[26];
            foreach (var x in s)
                frequency[x - 'a']++;
            var candidates = new List<char>(26);
            for (var i = 0; i < 26; i++)
            {
                if (frequency[i] < k)
                    continue;
                candidates.Add((char)(i + 'a'));
            }

            var queue = new Queue<string>();
            queue.Enqueue("");
            var result = "";
            while (queue.Count > 0)
            {
                var current = queue.Dequeue();
                foreach (var x in candidates)
                {
                    var next = current + x;
                    if (!Repeats(s, next, k))
                        continue;
                    queue.Enqueue(next);
                    result = next;
                }
            }

            return result;
        }

        public static bool Repeats(string s, string sub, int k)
        {
            var pointer = 0;
            var count = 0;
            foreach (var x in s)
            {
                if (x != sub[pointer])
                    continue;
                count++;
                pointer = 0;
                if (count == k)
                    return true;
            }

            return false;
        }

        public static int LongestSubsequence(string s, int k)
        {
            var current = 0;
            var count = 0;
            for (var i = s.Length - 1; i >= 0; i--)
            {
                var dx = (s[i] - '0') << count;
                if (current + dx > k)
                    continue;
                current += dx;
                count++;
            }

            for (var i = s.Length - 1; i >= 0; i--)
            {
                var dx = (s[i] - '0') << count;
                if (current + dx > k)
                    continue;
                current += dx;
                count++;
            }

            return count;
        }

        public static IList<int> FindKDistantIndices(int[] nums, int key, int k)
        {
            var result = new List<int>(nums.Length);
            var keyIndexes = new List<int>(nums.Length);
            for (var i = 0; i < nums.Length; i++)
            {
                if (nums[i] == key)
                    keyIndexes.Add(i);
            }

            for (var i = 0; i < nums.Length; i++)
            {
                foreach (var j in keyIndexes)
                {
                    if (Math.Abs(i - j) > k)
                        continue;
                    result.Add(i);
                    break;
                }
            }

            return result;
        }

        static bool IsPalindrome(long palindrome, int k)
        {
            var digits = new List<char>(64);
            var inBase = ToBase(palindrome, k);
            var x = inBase;
            if (x < 0)
                return false;
            while (x > 0)
            {
                digits.Add((char)(x % 10));
                x /= 10;
            }

            if (digits[0] == 0)
                return false;
            for (var j = 0; j < digits.Count / 2; j++)
                if (digits[j] != digits[digits.Count - j - 1])
                    return false;
            Console.WriteLine($"{palindrome} {inBase}");
            return true;
        }

        static long ToBase(long number, int baseK)
        {
            if (number == 0)
                return 0;

            var result = 0L;
            var place = 1;

            while (number > 0)
            {
                var digit = number % baseK;
                result += digit * place;
                number /= baseK;
                place *= 10;
            }

            return result;
        }

        public static string[] DivideString(string s, int k, char fill)
        {
            var result = new string[s.Length / k + (s.Length % k == 0 ? 0 : 1)];
            var sb = new StringBuilder(k);
            for (var i = 0; i < k * result.Length; i++)
            {
                var pointer = i / k;
                sb.Append(i < s.Length ? s[i] : fill);
                if (i % k == k - 1)
                {
                    result[pointer] = sb.ToString();
                    sb = new StringBuilder(k);
                }
            }

            return result;
        }

        public static int MinimumDeletions(string word, int k)
        {
            var freq = new int[26];
            foreach (var c in word)
                freq[c - 'a']++;
            var min = int.MaxValue;
            for (var i = 0; i < 26; i++)
            {
                var f = freq[i];
                if (f == 0)
                    continue;
                var toDelete = 0;
                for (var j = 0; j < 26; j++)
                {
                    if (j == i || freq[j] == 0)
                        continue;
                    if (freq[j] < f)
                        toDelete += freq[j];
                    else if (freq[j] > f + k)
                        toDelete += freq[j] - f - k;
                }

                min = Math.Min(min, toDelete);
            }

            return min;
        }

        public static int MaxDistance(string s, int k)
        {
            var width = 0;
            var height = 0;
            var max = 0;
            for (var i = 0; i < s.Length; i++)
            {
                var direction = s[i];
                if (direction == 'N')
                    height--;
                if (direction == 'S')
                    height++;
                if (direction == 'W')
                    width--;
                if (direction == 'E')
                    width++;
                max = Math.Max(
                    max, Math.Min(Math.Abs(width) + Math.Abs(height) + k * 2, i + 1));
            }

            return max;
        }

        public static int PartitionArray(int[] nums, int k)
        {
            var count = 0;
            Array.Sort(nums);
            var start = nums[0];
            for (var i = 1; i < nums.Length; i++)
            {
                if (nums[i] - start > k)
                {
                    start = nums[i];
                    count++;
                }
            }

            return count + 1;
        }

        public static int[][] DivideArray(int[] nums, int k)
        {
            Array.Sort(nums);
            var result = new int[nums.Length / 3][];
            for (var i = 0; i < nums.Length; i += 3)
            {
                var trio = new int[3];
                trio[0] = nums[i];
                for (var j = 1; j < 3; j++)
                {
                    trio[j] = nums[i + j];
                    if (trio[j] - trio[0] > k)
                        return Array.Empty<int[]>();
                }

                result[i / 3] = trio;
            }

            return result;
        }

        const int modulo = 1_000_000_007;
        const int x = 100000;
        static long[] factorial = new long[x];
        static long[] invariant = new long[x];

        long Power(long x, int n)
        {
            var result = 1L;
            while (n > 0)
            {
                if ((n & 1) == 1)
                    result = result * x % modulo;
                x = x * x % modulo;
                n >>= 1;
            }

            return result;
        }

        long Combine(int n, int m)
        {
            return factorial[n] * invariant[m] % modulo * invariant[n - m] % modulo;
        }

        public int CountGoodArrays(int n, int m, int k)
        {
            factorial[0] = 1;
            for (var i = 1; i < x; i++)
                factorial[i] = factorial[i - 1] * i % modulo;

            invariant[x - 1] = Power(factorial[x - 1], modulo - 2);
            for (var i = x - 1; i > 0; i--)
                invariant[i - 1] = invariant[i] * i % modulo;
            return (int)(Combine(n - 1, k) * m % modulo * Power(m - 1, n - k - 1) % modulo);
        }

        public static int[] NextGreaterElement(int[] nums1, int[] nums2)
        {
            var index = new Dictionary<int, int>(nums2.Length);
            for (var i = 0; i < nums2.Length; i++)
                index.Add(nums2[i], i);
            for (var p = 0; p < nums1.Length; p++)
            {
                var x = nums1[p];
                nums1[p] = -1;
                for (var i = index[x] + 1; i < nums2.Length; i++)
                    if (nums2[i] > x)
                    {
                        nums1[p] = nums2[i];
                        break;
                    }
            }

            return nums1;
        }

        public static int MaximumDifference(int[] nums)
        {
            var maxDiff = int.MinValue;
            var min = nums[0];
            for (var i = 1; i < nums.Length; i++)
            {
                var value = nums[i] - min;
                if (nums[i] > min && value > maxDiff)
                    maxDiff = value;
                min = Math.Min(min, nums[i]);
            }

            return maxDiff == int.MinValue ? -1 : maxDiff;
        }

        public static int MaxDiff(int num)
        {
            var reversedDigits = new List<int>(10);
            var x = num;
            while (x > 0)
            {
                reversedDigits.Add(x % 10);
                x /= 10;
            }

            var max = int.MinValue;
            var min = int.MaxValue;
            for (var i = 0; i < 10; i++)
            {
                var visited = new HashSet<int>(10);
                for (var j = 0; j < reversedDigits.Count; j++)
                {
                    if (!visited.Add(reversedDigits[j]))
                        continue;
                    if (i == 0 && reversedDigits.Count - 1 == j)
                        continue;
                    if (i == reversedDigits[j])
                    {
                        max = Math.Max(max, num);
                        min = Math.Min(min, num);
                        continue;
                    }

                    var newDigit = i;
                    var oldDigit = reversedDigits[j];
                    var power = 1;
                    var result = 0;
                    for (var k = 0; k < reversedDigits.Count; k++)
                    {
                        var t = reversedDigits[k];
                        if (newDigit == 0 && reversedDigits[^1] == oldDigit)
                            break;
                        if (t == oldDigit)
                            result += power * newDigit;
                        else
                            result += power * t;
                        power *= 10;
                    }

                    if (result != 0)
                    {
                        max = Math.Max(max, result);
                        min = Math.Min(min, result);
                    }
                }
            }

            return max - min;
        }

        public static int MinMaxDifference(int num)
        {
            var x = num;
            var digits = new List<int>(8);
            while (x > 0)
            {
                digits.Add(x % 10);
                x /= 10;
            }

            var pointer = digits.Count - 1;
            while (pointer >= 0 && digits[pointer] == 9)
            {
                pointer--;
            }

            if (pointer == -1)
                return num;
            var digitToMax = digits[pointer];
            var digitToMin = digits[^1];
            var power = 1;
            var max = 0;
            var min = 0;
            foreach (var t in digits)
            {
                if (t == digitToMax)
                    max += power * 9;
                else
                    max += power * t;

                if (t != digitToMin)
                    min += power * t;

                power *= 10;
            }

            return max - min;
        }

        public int MinimizeMax(int[] nums, int p)
        {
            Array.Sort(nums);
            var left = 0;
            var right = nums.Last() - nums.First();
            while (left < right)
            {
                var m = (left + right) / 2;
                if (CanForm(nums, p, m))
                    right = m;
                else
                    left = m + 1;
            }

            return left;
        }

        bool CanForm(int[] nums, int pairs, int middle)
        {
            var count = 0;
            for (var i = 0; i < nums.Length - 1 && count < pairs; i++)
            {
                if (nums[i + 1] - nums[i] >= middle)
                    continue;
                count++;
                i++;
            }

            return count >= pairs;
        }

        public static int MaxDifference(string s)
        {
            var frequency = new int[26];
            var maxOdd = 0;
            var minEven = int.MaxValue;
            foreach (var x in s)
                frequency[x - 'a']++;
            foreach (var f in frequency)
            {
                if (f != 0 && f % 2 == 0 && f < minEven)
                    minEven = f;
                else if (f % 2 == 1 && f > maxOdd)
                    maxOdd = f;
            }

            if (minEven == int.MaxValue)
                minEven = 0;
            return maxOdd - minEven;
        }

        public static int SumOddLengthSubarrays(int[] arr)
        {
            var answer = 0;
            for (var start = 0; start < arr.Length; start++)
            {
                var sum = arr[start];
                answer += sum;
                for (var end = start + 1; end < arr.Length; end++)
                {
                    sum += arr[end];
                    if ((end - start + 1) % 2 == 1)
                        answer += sum;
                }
            }

            return answer;
        }

        public static int SubarraySum(int[] nums)
        {
            var totalSum = 0;
            var total = new int [nums.Length];
            for (var i = 0; i < nums.Length; i++)
            {
                totalSum += nums[i];
                total[i] = totalSum;
            }

            totalSum = 0;

            for (var i = 0; i < nums.Length; i++)
            {
                var start = Math.Max(0, i - nums[i]);
                totalSum += total[i] - (start - 1 < 0 ? 0 : total[start - 1]);
            }

            return totalSum;
        }

        public int FindKthNumber(int n, int k)
        {
            var current = 1;
            k--;
            while (k > 0)
            {
                long steps = CountSteps(n, current, current + 1);
                if (steps <= n)
                {
                    current++;
                    k -= (int)steps;
                }
                else
                {
                    current *= 10;
                    k--;
                }
            }

            return current;
        }

        long CountSteps(int n, long current, long next)
        {
            var steps = 0L;
            while (current <= n)
            {
                steps += Math.Min(n + 1, next) - current;
                current *= 10;
                next *= 10;
            }

            return steps;
        }

        public IList<int> LexicalOrder(int n)
        {
            var ordered = new List<int>(n);
            var x = 1;
            for (var i = 1; i <= n; i++)
            {
                ordered.Add(x);
                if (x * 10 <= n)
                    x *= 10;
                else
                {
                    while (x % 10 == 9 || x >= n)
                        x /= 10;
                    x++;
                }
            }

            return ordered;
        }

        public static string ClearStars(string s)
        {
            var sb = new StringBuilder(s);
            var alphabet = new Stack<int>[26];
            for (var i = 0; i < s.Length; i++)
            {
                var x = s[i];
                if (x != '*')
                {
                    if (alphabet[x - 'a'] == null)
                        alphabet[x - 'a'] = new Stack<int>(s.Length);
                    alphabet[x - 'a'].Push(i);
                    continue;
                }

                for (var c = 'a'; c <= 'z'; c++)
                {
                    if (alphabet[c - 'a'] == null || alphabet[c - 'a'].Count == 0)
                        continue;
                    var index = alphabet[c - 'a'].Pop();
                    sb[i] = '\0';
                    sb[index] = '\0';
                    break;
                }
            }

            var result = new StringBuilder(s.Length);
            for (var i = 0; i < sb.Length; i++)
            {
                if (sb[i] != '\0')
                    result.Append(sb[i]);
            }

            return result.ToString();
        }

        public static string RobotWithString(string s)
        {
            var alphabet = new int[26];
            foreach (var c in s)
                alphabet[c - 'a']++;
            var robot = new Stack<char>(s.Length);
            var sb = new StringBuilder(s.Length);
            var minChar = 'a';
            foreach (var x in s)
            {
                robot.Push(x);
                alphabet[x - 'a']--;
                while (minChar != 'z' && alphabet[minChar - 'a'] == 0)
                    minChar++;
                while (robot.Count > 0 && robot.Peek() <= minChar)
                    sb.Append(robot.Pop());
            }

            return sb.ToString();
        }

        public static string SmallestEquivalentString(string s1, string s2, string baseStr)
        {
            var groups = new List<HashSet<char>>();
            for (var i = 0; i < s1.Length; i++)
            {
                var x = s1[i];
                var y = s2[i];
                var group1 =
                    groups
                        .Select((g, j) => new { Group = g, Index = j })
                        .SingleOrDefault(c => c.Group.Contains(x));
                var group2 =
                    groups
                        .Select((g, j) => new { Group = g, Index = j })
                        .SingleOrDefault(c => c.Group.Contains(y));

                if (group1 != null && group2 != null)
                {
                    if (group1.Index == group2.Index)
                        continue;
                    foreach (var g in group2.Group)
                        groups[group1.Index].Add(g);
                    groups.RemoveAt(group2.Index);
                }
                else if (group1 == null && group2 == null)
                {
                    groups.Add(new HashSet<char>());
                    groups[^1].Add(x);
                    groups[^1].Add(y);
                }
                else
                {
                    if (group1 == null)
                    {
                        groups[group2.Index].Add(x);
                    }
                    else
                        groups[group1.Index].Add(y);
                }
            }

            var dict = new Dictionary<char, int>();
            for (var i = 0; i < groups.Count; i++)
            {
                foreach (var c in groups[i])
                    dict.Add(c, i);
            }

            var sb = new StringBuilder();
            foreach (var c in baseStr)
            {
                if (!dict.ContainsKey(c))
                {
                    sb.Append(c);
                    continue;
                }

                var group = dict[c];
                var min = groups[group].OrderBy(x => x).First();
                sb.Append(min);
            }

            return sb.ToString();
        }

        public static int MaxRepeating(string sequence, string word)
        {
            var max = 0;
            var p = 0;
            var current = 0;
            for (var i = 0; i < sequence.Length - word.Length + 1; i++)
            {
                if (sequence[i] != word[p])
                {
                    p = 0;
                    current = 0;
                }
                else
                {
                    p++;
                    if (p == word.Length)
                    {
                        current++;
                        max = Math.Max(max, current);
                        p = 0;
                    }
                }
            }

            return max;
        }

        public static string AnswerString(string word, int numFriends)
        {
            if (numFriends == 1)
                return word;
            var lexicographic = string.Empty;
            var length = word.Length - numFriends + 1;
            for (var i = 0; i + length - 1 < word.Length; i++)
            {
                var w = word.Substring(i, Math.Min(length, word.Length - i));
                if (string.Compare(w, lexicographic, StringComparison.Ordinal) > 0)
                    lexicographic = w;
            }

            return lexicographic;
        }

        public int MaxCandies(int[] status, int[] candies, int[][] keys, int[][] containedBoxes, int[] initialBoxes)
        {
            var n = status.Length;
            var hasKey = new bool[n];
            var boxOwned = new bool[n];
            var boxVisited = new bool[n];

            var queue = new Queue<int>();

            foreach (var box in initialBoxes)
            {
                boxOwned[box] = true;
                queue.Enqueue(box);
            }

            var totalCandies = 0;
            var progress = true;

            while (progress)
            {
                progress = false;
                var size = queue.Count;

                for (var i = 0; i < size; i++)
                {
                    var box = queue.Dequeue();
                    if (!boxVisited[box] && (status[box] == 1 || hasKey[box]))
                    {
                        boxVisited[box] = true;
                        progress = true;

                        totalCandies += candies[box];

                        foreach (var key in keys[box])
                        {
                            hasKey[key] = true;
                            if (boxOwned[key] && !boxVisited[key])
                            {
                                queue.Enqueue(key);
                            }
                        }

                        foreach (var contained in containedBoxes[box])
                        {
                            boxOwned[contained] = true;
                            queue.Enqueue(contained);
                        }
                    }
                    else
                    {
                        queue.Enqueue(box);
                    }
                }
            }

            return totalCandies;
        }

        public static int Candy(int[] ratings)
        {
            var n = ratings.Length;
            var candies = new int[n];
            Array.Fill(candies, 1);
            for (var i = 1; i < n; i++)
            {
                if (ratings[i] > ratings[i - 1])
                    candies[i] = candies[i - 1] + 1;
            }

            for (var i = n - 1; i > 0; i--)
            {
                if (candies[i - 1] > ratings[i])
                    candies[i - 1] = Math.Max(candies[i] + 1, candies[i - 1]);
            }

            return candies.Sum();
        }

        public static long DistributeCandies(int n, int limit)
        {
            return Distribure(n + 2) - 3 * Distribure(n - limit + 1) +
                3 * Distribure(n - (limit + 1) * 2 + 2) - Distribure(n - 3 * (limit + 1) + 2);
        }

        static long Distribure(int n)
        {
            if (n < 0)
                return 0;
            return (long)n * (n - 1) / 2;
        }

        public static int SnakesAndLadders(int[][] board)
        {
            var n = board.Length;
            var magic = new Dictionary<int, int>(n * n);
            for (var y = 0; y < n; y++)
            {
                var yi = n - 1 - y;
                var isReversed = (yi + 1) % 2 == 0;
                for (var x = 0; x < n; x++)
                {
                    if (board[y][x] == -1)
                        continue;
                    var value = yi * n;
                    if (isReversed)
                        value += n - x;
                    else
                        value += x + 1;
                    magic.Add(value, board[y][x]);
                }
            }

            var queue = new Queue<(int Node, int Steps)>(n * n);
            queue.Enqueue((1, 0));
            var visited = new HashSet<int>(n * n);
            var min = int.MaxValue;
            while (queue.Count > 0)
            {
                var (node, steps) = queue.Dequeue();
                if (node == n * n)
                {
                    min = Math.Min(min, steps);
                    continue;
                }

                if (!visited.Add(node))
                    continue;
                for (var next = node + 1; next <= Math.Min(node + 6, n * n); next++)
                {
                    if (magic.ContainsKey(next))
                        queue.Enqueue((magic[next], steps + 1));
                    else
                        queue.Enqueue((next, steps + 1));
                }
            }

            return min == int.MaxValue ? -1 : min;
        }

        public int ClosestMeetingNode(int[] edges, int node1, int node2)
        {
            var min = int.MaxValue;
            var index = -1;
            var path1 = Search(edges, node1);
            var path2 = Search(edges, node2);
            for (var i = 0; i < edges.Length; i++)
            {
                if (path1[i] == -1 || path2[i] == -1)
                    continue;
                var x = Math.Max(path1[i], path2[i]);
                if (x < min)
                {
                    min = x;
                    index = i;
                }
            }

            return index;
        }

        public int[] Search(int[] edges, int start)
        {
            var n = edges.Length;
            var visited = new HashSet<int>(n);
            var queue = new Queue<(int Node, int Length)>(n);
            queue.Enqueue((start, 0));
            var path = new int[n];
            Array.Fill(path, -1);
            while (queue.Count > 0)
            {
                var (node, length) = queue.Dequeue();
                if (!visited.Add(node))
                    continue;
                path[node] = Math.Max(path[node], length);
                if (edges[node] == -1)
                    continue;
                queue.Enqueue((edges[node], length + 1));
            }

            return path;
        }

        public static bool HasMatch(string s, string p)
        {
            var parts = p.Split('*');
            var prefix = parts[0];
            var suffix = parts[1];
            var pointer = 0;
            var end = 0;
            for (var i = 0; i < s.Length; i++)
            {
                if (s[i] != prefix[pointer])
                {
                    pointer = 0;
                }

                pointer++;
                if (pointer == prefix.Length)
                {
                    end = i;
                    break;
                }
            }

            if (s.Length - end + 1 < suffix.Length || pointer < prefix.Length)
                return false;
            pointer = 0;
            for (var i = end + 1; i < s.Length; i++)
            {
                if (s[i] != suffix[pointer])
                {
                    pointer = 0;
                }

                pointer++;
                if (pointer == suffix.Length)
                    break;
            }

            return pointer == suffix.Length;
        }

        public int[] MaxTargetNodes(int[][] edges1, int[][] edges2, int k)
        {
            var n = edges1.Length + 1;
            var m = edges2.Length + 1;
            var dict1 = GetNeighbours(edges1);
            var dict2 = GetNeighbours(edges2);
            var max = 0;
            for (var e = 0; e < m; e++)
                max = Math.Max(max, DepthSearch(dict2, e, -1, k - 1));
            var result = new int[n];
            for (var e = 0; e < n; e++)
            {
                result[e] = max;
                result[e] += DepthSearch(dict1, e, -1, k);
            }

            return result;
        }

        Dictionary<int, List<int>> GetNeighbours(int[][] edges)
        {
            var dict = new Dictionary<int, List<int>>();
            foreach (var e in edges)
            {
                dict.TryAdd(e[0], new List<int>());
                dict[e[0]].Add(e[1]);
                dict.TryAdd(e[1], new List<int>());
                dict[e[1]].Add(e[0]);
            }

            return dict;
        }

        int DepthSearch(Dictionary<int, List<int>> edges, int edge, int parent, int maxDepth)
        {
            if (maxDepth < 0)
                return 0;
            if (maxDepth == 0)
                return 1;
            var depth = 1;
            if (!edges.ContainsKey(edge))
                return 0;
            foreach (var node in edges[edge])
            {
                if (node == parent)
                    continue;
                depth += DepthSearch(edges, node, edge, maxDepth - 1);
            }

            return depth;
        }

        public int DifferenceOfSums(int n, int m)
        {
            var sum = 0;
            for (var i = 1; i <= n; i++)
            {
                if (i % m != 0)
                    sum += i;
                else
                    sum -= i;
            }

            return sum;
        }

        public int LargestPathValue(string colors, int[][] edges)
        {
            var n = colors.Length;
            var dict = new Dictionary<int, List<int>>();
            var inDegree = new int[n];

            foreach (var edge in edges)
            {
                if (!dict.ContainsKey(edge[0]))
                    dict.Add(edge[0], new List<int>());
                dict[edge[0]].Add(edge[1]);
                inDegree[edge[1]]++;
            }

            var queue = new Queue<int>();
            for (var i = 0; i < n; i++)
            {
                if (inDegree[i] == 0)
                    queue.Enqueue(i);
            }

            var frequency = new int[n, 26];
            var visited = 0;
            var result = 0;

            while (queue.Count > 0)
            {
                var node = queue.Dequeue();
                visited++;
                var colorIndex = colors[node] - 'a';
                frequency[node, colorIndex]++;
                result = Math.Max(result, frequency[node, colorIndex]);
                if (!dict.ContainsKey(node))
                    continue;
                foreach (var e in dict[node])
                {
                    for (var i = 0; i < 26; i++)
                        frequency[e, i] = Math.Max(frequency[e, i], frequency[node, i]);
                    inDegree[e]--;
                    if (inDegree[e] == 0)
                        queue.Enqueue(e);
                }
            }

            return visited == n ? result : -1;
        }

        public static int LongestPalindrome(string[] words)
        {
            var pairs = new Dictionary<(char X, char Y), int>();
            foreach (var w in words)
            {
                if (!pairs.ContainsKey((w[0], w[1])))
                    pairs.Add((w[0], w[1]), 0);
                pairs[(w[0], w[1])]++;
            }

            var length = 0;
            foreach (var pair in pairs.Keys)
            {
                if (pair == (pair.Y, pair.X))
                {
                    length += 2 * (pairs[pair] - pairs[pair] % 2);
                    continue;
                }

                if (!pairs.ContainsKey((pair.Y, pair.X)))
                    continue;
                length += 4 * Math.Min(pairs[pair], pairs[(pair.Y, pair.X)]);
                pairs.Remove(pair);
                pairs.Remove((pair.Y, pair.X));
            }

            var oddTwins = pairs
                .Keys
                .Where(x => x.X == x.Y)
                .Where(x => pairs[x] % 2 == 1)
                .ToArray();
            if (oddTwins.Any())
                length += 2;
            return length;
        }

        public IList<int> FindWordsContaining(string[] words, char x)
        {
            var list = new List<int>(words.Length);
            for (var i = 0; i < words.Length; i++)
            {
                if (words[i].Contains(x))
                    list.Add(i);
            }

            return list;
        }

        public long MaximumValueSum(int[] nums, int k, int[][] edges)
        {
            var sum = 0L;
            var count = 0;
            var minLoss = int.MaxValue;
            foreach (var x in nums)
            {
                var xor = x ^ k;
                sum += Math.Max(xor, x);
                if (xor > x)
                    count++;
                minLoss = Math.Min(minLoss, Math.Abs(x - xor));
            }

            if (count % 2 == 1)
                sum -= minLoss;
            return sum;
        }

        public int MaxRemoval(int[] nums, int[][] queries)
        {
            Array.Sort(queries, (a, b) => a[0] - b[0]);
            var pq = new PriorityQueue<int, int>(
                Comparer<int>.Create((a, b) => -1 * a.CompareTo(b)));
            var sweep = new int[nums.Length + 1];
            var index = 0;
            var cs = 0;
            var taken = 0;
            for (var i = 0; i < nums.Length; i++)
            {
                cs += sweep[i];
                while (index < queries.Length && queries[index][0] <= i)
                {
                    pq.Enqueue(queries[index][1], queries[index][1]);
                    index++;
                }

                while (pq.Count > 0 && cs < nums[i])
                {
                    var top = pq.Peek();
                    pq.Dequeue();
                    if (top < i)
                    {
                        continue;
                    }

                    cs++;
                    taken++;
                    sweep[top + 1] += -1;
                }

                if (cs < nums[i])
                    return -1;
            }

            return queries.Length - taken;
        }

        public int HammingDistance(int x, int y)
        {
            var distance = 0;
            var xor = x ^ y;
            for (var i = 0; i < 31; i++)
                distance += (xor & (1 << i)) >> i;
            return distance;
        }

        public void SetZeroes(int[][] matrix)
        {
            var height = matrix.Length;
            var width = matrix[0].Length;
            var rows = new HashSet<int>();
            var columns = new HashSet<int>();
            for (var i = 0; i < height; i++)
            {
                for (var j = 0; j < width; j++)
                {
                    if (matrix[i][j] == 0)
                    {
                        rows.Add(i);
                        columns.Add(j);
                    }
                }
            }

            for (var i = 0; i < height; i++)
            {
                for (var j = 0; j < width; j++)
                {
                    if (rows.Contains(i) || columns.Contains(j))
                        matrix[i][j] = 0;
                }
            }
        }

        public static bool IsZeroArray(int[] nums, int[][] queries)
        {
            var difference = new int[nums.Length + 1];
            difference[0] = nums[0];
            for (var i = 1; i < nums.Length; i++)
            {
                difference[i] = nums[i] - nums[i - 1];
            }

            foreach (var query in queries)
            {
                var start = query[0];
                var end = query[1];
                difference[start]--;
                difference[end + 1]++;
            }

            nums[0] = difference[0];
            for (var i = 1; i < nums.Length; i++)
                nums[i] = nums[i - 1] + difference[i];

            return !nums.Any(x => x > 0);
        }

        public string TriangleType(int[] nums)
        {
            if (nums[0] > nums[1])
                (nums[0], nums[1]) = (nums[1], nums[0]);
            if (nums[1] > nums[2])
                (nums[2], nums[1]) = (nums[2], nums[1]);
            if (nums[0] > nums[1])
                (nums[0], nums[1]) = (nums[0], nums[1]);
            var a = nums[0];
            var b = nums[1];
            var c = nums[2];
            if (a > b + c)
                return "none";
            if (a == b && b == c)
                return "equilateral";
            if (a == b || b == c || c == a)
                return "isosceles";
            return "scalene";
        }

        public int ColorTheGrid(int m, int n)
        {
            const int modulo = 1000000007;
            var valid = new Dictionary<int, List<int>>();
            var maskEnd = (int)Math.Pow(3, m);
            for (var mask = 0; mask < maskEnd; ++mask)
            {
                var color = new List<int>();
                var x = mask;
                for (var i = 0; i < m; ++i)
                {
                    color.Add(x % 3);
                    x /= 3;
                }

                var check = true;
                for (var i = 0; i < m - 1; ++i)
                {
                    if (color[i] != color[i + 1])
                        continue;
                    check = false;
                    break;
                }

                if (check)
                    valid[mask] = color;
            }

            var adjacent = new Dictionary<int, List<int>>();
            foreach (var mask1 in valid.Keys)
            foreach (var mask2 in valid.Keys)
            {
                var check = true;
                for (var i = 0; i < m; ++i)
                {
                    if (valid[mask1][i] != valid[mask2][i])
                        continue;
                    check = false;
                    break;
                }

                if (!check)
                    continue;
                if (!adjacent.ContainsKey(mask1))
                    adjacent[mask1] = new List<int>();
                adjacent[mask1].Add(mask2);
            }

            var f = new Dictionary<int, int>();
            foreach (var mask in valid.Keys)
                f[mask] = 1;

            for (var i = 1; i < n; ++i)
            {
                var g = new Dictionary<int, int>();
                foreach (var mask2 in valid.Keys)
                {
                    if (!adjacent.ContainsKey(mask2))
                        continue;
                    foreach (var mask1 in adjacent[mask2])
                    {
                        g.TryAdd(mask2, 0);
                        g[mask2] = (g[mask2] + f[mask1]) % modulo;
                    }
                }

                f = g;
            }

            return f.Values.Aggregate(0, (current, num) => (current + num) % modulo);
        }

        public static void SortColors(int[] nums)
        {
            var index = 0;
            for (var i = 0; i < nums.Length; i++)
            {
                if (nums[i] == 0)
                {
                    (nums[index], nums[i]) = (nums[i], nums[index]);
                    index++;
                }
            }

            for (var i = 0; i < nums.Length; i++)
            {
                if (nums[i] == 1)
                {
                    (nums[index], nums[i]) = (nums[i], nums[index]);
                    index++;
                }
            }

            for (var i = 0; i < nums.Length; i++)
            {
                if (nums[i] == 2)
                {
                    (nums[index], nums[i]) = (nums[i], nums[index]);
                    index++;
                }
            }
        }

        public static IList<string> GetWordsInLongestSubsequence(string[] words, int[] groups)
        {
            var n = groups.Length;
            var dp = new int[n];
            var prev = new int[n];
            Array.Fill(dp, 1);
            Array.Fill(prev, -1);
            var maxIndex = 0;

            for (var i = 1; i < n; i++)
            {
                for (var j = 0; j < i; j++)
                {
                    if (GetDifferenceIndex(words[i], words[j]) == -1 || dp[j] + 1 <= dp[i] ||
                        groups[i] == groups[j])
                        continue;
                    dp[i] = dp[j] + 1;
                    prev[i] = j;
                }

                if (dp[i] > dp[maxIndex])
                    maxIndex = i;
            }

            var result = new List<string>();
            for (var i = maxIndex; i >= 0; i = prev[i])
                result.Add(words[i]);

            result.Reverse();
            return result;
        }

        static int GetDifferenceIndex(string source, string target)
        {
            if (source.Length != target.Length)
                return -1;
            var index = -1;
            for (var i = 0; i < source.Length; i++)
            {
                if (source[i] != target[i])
                {
                    if (index != -1)
                        return -1;
                    index = i;
                }
            }

            return index;
        }

        public IList<string> GetLongestSubsequence(string[] words, int[] groups)
        {
            var fromZero = new List<int>(words.Length);
            var query = 0;
            for (var i = 0; i < groups.Length; i++)
            {
                if (groups[i] != query)
                    continue;
                fromZero.Add(i);
                query = Math.Abs(query - 1);
            }

            query = 1;
            var fromOne = new List<int>(words.Length);
            for (var i = 0; i < groups.Length; i++)
            {
                if (groups[i] != query)
                    continue;
                fromOne.Add(i);
                query = Math.Abs(query - 1);
            }

            if (fromOne.Count > fromZero.Count)
            {
                return fromOne.Select(i => words[i]).ToList();
            }

            return fromZero.Select(i => words[i]).ToList();
        }

        private const int MOD = 1_000_000_007;

        public int LengthAfterTransformations(string s, int t, IList<int> nums)
        {
            var transition = new int[26, 26];
            for (var i = 0; i < 26; i++)
            for (var j = 1; j <= nums[i]; j++)
            {
                var nextChar = (i + j) % 26;
                transition[i, nextChar]++;
            }

            var resultMatrix = MatrixPower(transition, t);
            var finalCounts = new int[26];
            for (var i = 0; i < 26; i++)
            for (var j = 0; j < 26; j++)
            {
                finalCounts[i] = (finalCounts[i] + resultMatrix[i, j]) % modulo;
            }

            var total = 0L;
            foreach (var ch in s)
            {
                total = (total + finalCounts[ch - 'a']) % modulo;
            }

            return (int)total;
        }

        private int[,] MatrixPower(int[,] matrix, int power)
        {
            var result = new int[26, 26];
            for (var i = 0; i < 26; i++)
                result[i, i] = 1;
            while (power > 0)
            {
                if ((power & 1) == 1)
                    result = MultiplyMatrices(result, matrix);

                matrix = MultiplyMatrices(matrix, matrix);
                power >>= 1;
            }

            return result;
        }

        private int[,] MultiplyMatrices(int[,] a, int[,] b)
        {
            var res = new int[26, 26];

            for (var i = 0; i < 26; i++)
            for (var k = 0; k < 26; k++)
            for (var j = 0; j < 26; j++)
                res[i, j] = (int)((res[i, j] + 1L * a[i, k] * b[k, j]) % modulo);

            return res;
        }

        public static int LengthAfterTransformations(string source, int t)
        {
            var module = 1000000007;
            var frequency = new long[26];
            var count = 0L;

            foreach (var x in source)
                frequency[x - 'a']++;

            for (var s = 0; s < t; s++)
            {
                var newFrequency = new long[26];
                for (var i = 0; i < 25; i++)
                    newFrequency[i + 1] = (frequency[i] + newFrequency[i + 1]) % module;
                newFrequency[0] = (frequency[25] + newFrequency[0]) % module;
                newFrequency[1] = (frequency[25] + newFrequency[1]) % module;
                frequency = newFrequency;
            }


            foreach (var f in frequency)
                count = (count + f) % module;

            return (int)count;
        }

        public static int[] FindEvenNumbers(int[] digits)
        {
            var frequency = new int[10];
            var max = 0;
            foreach (var x in digits)
            {
                frequency[x]++;
                max = Math.Max(max, x);
            }

            var maxCount = 98 * 99 * 100 / 2 / 3;
            var result = new List<int>(maxCount);
            var visited = new HashSet<int>(maxCount);
            for (var i = 0; i <= max; i++)
            {
                if (frequency[i] == 0 || i == 0)
                    continue;
                frequency[i]--;
                for (var j = 0; j <= max; j++)
                {
                    if (frequency[j] == 0)
                        continue;
                    frequency[j]--;
                    foreach (var z in new[] { 0, 2, 4, 6, 8 })
                    {
                        if (frequency[z] == 0)
                            continue;
                        var x = z + 10 * j + 100 * i;
                        if (!visited.Add(x))
                            continue;
                        result.Add(x);
                    }

                    frequency[j]++;
                }

                frequency[i]++;
            }

            return result.ToArray();
        }

        public static bool ThreeConsecutiveOdds(int[] arr)
        {
            var odds = 0;
            foreach (var x in arr)
            {
                if (x % 2 == 0)
                {
                    odds = 0;
                    continue;
                }

                odds++;

                if (odds == 3)
                    return true;
            }

            return false;
        }

        public static long MinSum(int[] nums1, int[] nums2)
        {
            var sum1 = 0L;
            var sum1WithReplacing = 0L;
            var sum2 = 0L;
            var sum2WithReplacing = 0L;
            foreach (var x in nums1)
            {
                sum1 += x;
                sum1WithReplacing += x;
                if (x == 0)
                    sum1WithReplacing++;
            }

            foreach (var x in nums2)
            {
                sum2 += x;
                sum2WithReplacing += x;
                if (x == 0)
                    sum2WithReplacing++;
            }

            if (sum1 == sum1WithReplacing && sum1 < sum2WithReplacing)
                return -1;

            if (sum2 == sum2WithReplacing && sum2 < sum1WithReplacing)
                return -1;

            return Math.Max(sum1WithReplacing, sum2WithReplacing);
        }

        public int MinTimeToReach2(int[][] moveTime)
        {
            var height = moveTime.Length;
            var width = moveTime[0].Length;
            var queue = new PriorityQueue<(int X, int Y, int Time, bool IsFirst), int>();
            queue.Enqueue((0, 0, 0, true), moveTime[0][0]);
            var visited = new HashSet<(int X, int Y)>(height * width);
            while (queue.Count > 0)
            {
                var (x, y, t, isFirst) = queue.Dequeue();
                var dt = 1;
                if (!isFirst)
                    dt = 2;
                if (x == width - 1 && y == height - 1)
                    return t;
                if (!visited.Add((x, y)))
                    continue;
                if (x - 1 >= 0)
                {
                    var ti = moveTime[y][x - 1];
                    var newTime = Math.Max(ti, t) + dt;
                    if (!visited.Contains((x - 1, y)))
                        queue.Enqueue((x - 1, y, newTime, !isFirst), newTime);
                }

                if (x + 1 < width)
                {
                    var ti = moveTime[y][x + 1];
                    var newTime = Math.Max(ti, t) + dt;
                    if (!visited.Contains((x + 1, y)))
                        queue.Enqueue((x + 1, y, newTime, !isFirst), newTime);
                }

                if (y - 1 >= 0)
                {
                    var ti = moveTime[y - 1][x];
                    var newTime = Math.Max(ti, t) + dt;
                    if (!visited.Contains((x, y - 1)))
                        queue.Enqueue((x, y - 1, newTime, !isFirst), newTime);
                }

                if (y + 1 < height)
                {
                    var ti = moveTime[y + 1][x];
                    var newTime = Math.Max(ti, t) + dt;
                    if (!visited.Contains((x, y + 1)))
                        queue.Enqueue((x, y + 1, newTime, !isFirst), newTime);
                }
            }

            return -1;
        }

        public int MinTimeToReach(int[][] moveTime)
        {
            var height = moveTime.Length;
            var width = moveTime[0].Length;
            var queue = new PriorityQueue<(int X, int Y, int Time), int>();
            queue.Enqueue((0, 0, 0), moveTime[0][0]);
            var visited = new HashSet<(int X, int Y)>(height * width);
            while (queue.Count > 0)
            {
                var (x, y, t) = queue.Dequeue();
                if (x == width - 1 && y == height - 1)
                    return t;
                if (!visited.Add((x, y)))
                    continue;
                if (x - 1 >= 0)
                {
                    var ti = moveTime[y][x - 1];
                    var newTime = Math.Max(ti, t) + 1;
                    if (!visited.Contains((x - 1, y)))
                        queue.Enqueue((x - 1, y, newTime), newTime);
                }

                if (x + 1 < width)
                {
                    var ti = moveTime[y][x + 1];
                    var newTime = Math.Max(ti, t) + 1;
                    if (!visited.Contains((x + 1, y)))
                        queue.Enqueue((x + 1, y, newTime), newTime);
                }

                if (y - 1 >= 0)
                {
                    var ti = moveTime[y - 1][x];
                    var newTime = Math.Max(ti, t) + 1;
                    if (!visited.Contains((x, y - 1)))
                        queue.Enqueue((x, y - 1, newTime), newTime);
                }

                if (y + 1 < height)
                {
                    var ti = moveTime[y + 1][x];
                    var newTime = Math.Max(ti, t) + 1;
                    if (!visited.Contains((x, y + 1)))
                        queue.Enqueue((x, y + 1, newTime), newTime);
                }
            }

            return -1;
        }

        public int[] BuildArray(int[] nums)
        {
            var ans = new int[nums.Length];
            for (var i = 0; i < nums.Length; i++)
                ans[i] = nums[nums[i]];
            return ans;
        }

        public static int NumTilings(int n)
        {
            var dp = new int[n < 4 ? 4 : n + 1];
            dp[1] = 1;
            dp[2] = 2;
            dp[3] = 5;
            if (n < 4)
                return dp[n];
            var modulo = 1_000_000_007;
            for (var i = 4; i <= n; i++)
            {
                dp[i] = (2 * dp[i - 1] % modulo + dp[i - 3] % modulo) % modulo;
            }

            return dp[n];
        }

        public int NumEquivDominoPairs(int[][] dominoes)
        {
            var dict = new Dictionary<int, int>(dominoes.Length);
            foreach (var x in dominoes)
            {
                var value = 10 * Math.Min(x[0], x[1]) + Math.Max(x[0], x[1]);
                dict.TryAdd(value, 0);
                dict[value]++;
            }

            var count = 0;
            foreach (var x in dict.Values)
            {
                if (x > 1)
                    count += x * (x - 1) / 2;
            }

            return count;
        }

        public int MinDominoRotations(int[] tops, int[] bottoms)
        {
            var a = tops[0];
            var b = bottoms[0];
            var n = tops.Length;
            var c = tops[n - 1];
            var d = bottoms[n - 1];
            var array = new[] { a, b, c, d }.Distinct().ToArray();
            var list = new List<int>();
            for (var i = 0; i < array.Length; i++)
            {
                list.Add(Meow(tops, bottoms, array[i]));
                list.Add(Meow(bottoms, tops, array[i]));
            }

            if (list.Any(x => x >= 0))
                return list.Where(x => x >= 0).Min();
            return -1;
        }

        int Meow(int[] tops, int[] bottoms, int a)
        {
            var n = tops.Length;
            var topsCount = 0;
            for (var i = 0; i < n; i++)
            {
                if (tops[i] == a)
                    continue;
                if (bottoms[i] != a)
                    return -1;
                topsCount++;
            }

            return Math.Min(topsCount, n - topsCount);
        }

        public static string PushDominoes(string dominoes)
        {
            var sb = new StringBuilder(dominoes);
            for (var i = 0; i < dominoes.Length - 1; i++)
            {
                if (dominoes[i] != '.' && dominoes[i + 1] == '.')
                {
                    var right = i + 1;
                    while (right < dominoes.Length && dominoes[right] == '.')
                        right++;
                    if (right >= dominoes.Length)
                        continue;
                    var dots = right - i - 1;
                    if (dominoes[i] == dominoes[right])
                    {
                        for (var j = 0; j < dots; j++)
                            sb[i + 1 + j] = dominoes[i];
                    }

                    if (dominoes[i] == 'R' && dominoes[right] == 'L')
                    {
                        for (var j = 0; j < dots / 2; j++)
                            sb[i + 1 + j] = 'R';
                        for (var j = 0; j < dots / 2; j++)
                            sb[right - 1 - j] = 'L';
                    }

                    i = right - 1;
                }
            }

            var indexOfR = dominoes.IndexOf('R');
            if (indexOfR == -1)
                indexOfR = dominoes.Length;
            for (var i = indexOfR - 1; i > 0; i--)
            {
                if (sb[i] == 'L' && sb[i - 1] == '.')
                    sb[i - 1] = 'L';
            }

            var indexOfL = dominoes.LastIndexOf('L');
            for (var i = indexOfL + 1; i < dominoes.Length - 1; i++)
            {
                if (sb[i] == 'R' && sb[i + 1] == '.')
                    sb[i + 1] = 'R';
            }

            return sb.ToString();
        }

        public static string ReverseStr(string s, int k)
        {
            var sb = new StringBuilder(s);
            for (var i = Math.Min(k - 1, s.Length - 1); i < s.Length; i += 2 * k)
            {
                for (var j = 0; j < k / 2; j++)
                    (sb[Math.Max(0, Math.Min(i - k + 1 + j, s.Length - 1))], sb[i - j]) = (sb[i - j],
                        sb[Math.Max(0, Math.Min(i - k + 1 + j, s.Length - 1))]);
            }

            return sb.ToString();
        }

        public static int MaxTaskAssign(int[] tasksInput, int[] workers, int pills, int strength)
        {
            var done = 0;
            var tasks = new SortedSet<int>();
            var dict = new Dictionary<int, int>();
            var failedDict = new Dictionary<int, int>();
            foreach (var x in tasksInput)
            {
                tasks.Add(x);
                dict.TryAdd(x, 0);
                dict[x]++;
            }

            var failedTasks = new SortedSet<int>();
            Array.Sort(workers);
            var workersPointer = workers.Length - 1;
            while (workersPointer >= 0 && tasks.Count > 0)
            {
                var worker = workers[workersPointer];
                var task = GetLeftBorder(tasks, worker, 0, tasks.Count);
                workersPointer--;
                dict[task]--;
                if (dict[task] == 0)
                {
                    tasks.Remove(task);
                    dict.Remove(task);
                }

                if (worker >= task)
                    done++;
                else
                {
                    failedTasks.Add(task);
                    failedDict.TryAdd(task, 0);
                    failedDict[task]++;
                    workersPointer++;
                }
            }

            while (failedTasks.Count > 0 && pills > 0)
            {
                var worker = workers[workersPointer] + strength;
                pills--;
                var task = GetLeftBorder(failedTasks, worker, 0, failedTasks.Count);
                workersPointer--;
                failedDict[task]--;
                if (failedDict[task] == 0)
                {
                    failedDict.Remove(task);
                    failedTasks.Remove(task);
                }

                if (worker >= task)
                    done++;
                else
                    workersPointer++;
            }

            return done;
        }

        public static int GetLeftBorder(SortedSet<int> tasks, int strength, int left, int right)
        {
            if (left == right - 1)
                return tasks.ElementAt(left);
            var m = left + (right - left) / 2;
            var task = tasks.ElementAt(m);
            if (task > strength)
                return GetLeftBorder(tasks, strength, left, m);
            if (task == strength)
                return task;
            return GetLeftBorder(tasks, strength, m, right);
        }

        public int FindNumbers(int[] nums)
        {
            var count = 0;
            for (var i = 0; i < nums.Length; i++)
            {
                var x = nums[i];
                var digits = 0;
                while (x > 0)
                {
                    digits++;
                    x /= 10;
                }

                if (digits % 2 == 0)
                    count++;
            }

            return count;
        }

        public static long CountSubarrays(int[] nums, int k)
        {
            var max = nums.Max();
            var left = 0;
            var count = 0;
            var result = 0L;
            for (var right = 0; right < nums.Length; right++)
            {
                if (nums[right] == max)
                    count++;
                if (count == k)
                {
                    while (left <= right && count >= k)
                    {
                        result += nums.Length - right;
                        if (nums[left] == max)
                            count--;
                        left++;
                    }
                }
            }

            return result;
        }

        public static long CountSubarrays1(int[] nums, long k)
        {
            var sum = 0L;
            var result = 0L;
            var left = 0;
            for (var right = 0; right < nums.Length; right++)
            {
                sum += nums[right];
                while (left <= right && sum * (right - left + 1) >= k)
                {
                    sum -= nums[left];
                    left++;
                }

                result += right - left + 1;
            }

            return result;
        }

        public int CountSubarrays(int[] nums)
        {
            var result = 0;
            for (var i = 0; i < nums.Length - 2; i++)
            {
                if (2 * (nums[i] + nums[i + 2]) == nums[i + 1])
                    result++;
            }

            return result;
        }

        public static long CountSubarrays(int[] nums, int min, int max)
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

        public static long CountInterestingSubarrays(IList<int> nums, int modulo, int k)
        {
            var dict = new Dictionary<int, int>();
            dict.Add(0, 1);
            var result = 0L;
            var current = 0;
            foreach (var x in nums)
            {
                if (x % modulo == k)
                    current++;
                var target = (current - k + modulo) % modulo;
                if (dict.ContainsKey(target))
                    result += dict[target];
                dict[current % modulo] = dict.GetValueOrDefault(current % modulo, 0) + 1;
            }

            return result;
        }

        public static int CountCompleteSubarrays(int[] nums)
        {
            var distinctCount = nums.Distinct().Count();
            var count = 0;
            var dict = new Dictionary<int, int>(distinctCount);
            var left = 0;
            for (var right = 0; right < nums.Length; right++)
            {
                var value = nums[right];
                dict.TryAdd(value, 0);
                dict[value]++;
                if (dict.Keys.Count != distinctCount)
                {
                    continue;
                }

                while (left <= right && dict.Keys.Count == distinctCount)
                {
                    count += nums.Length - right;
                    var toDelete = nums[left];
                    dict[toDelete]--;
                    if (dict[toDelete] == 0)
                        dict.Remove(toDelete);
                    left++;
                }
            }

            return count;
        }

        public static int CountLargestGroup(int n)
        {
            var maxCount = 0;
            var dict = new Dictionary<int, int>(n);
            for (var i = 1; i <= n; i++)
            {
                var sum = 0;
                var x = i;
                while (x > 0)
                {
                    sum += x % 10;
                    x /= 10;
                }

                dict.TryAdd(x, 0);
                dict[sum]++;
                maxCount = Math.Max(maxCount, dict[sum]);
            }

            var count = 0;
            foreach (var x in dict.Values)
            {
                if (x == maxCount)
                    count++;
            }

            return count;
        }

        public static int NumberOfArrays(int[] differences, int lower, int upper)
        {
            var n = differences.Length;
            var array = new long[n + 1];
            var min = 0L;
            var max = 0L;
            for (var i = array.Length - 2; i >= 0; i--)
            {
                array[i] = array[i + 1] - differences[i];
                if (array[i] < min)
                    min = array[i];
                if (array[i] > max)
                    max = array[i];
            }

            var diff = max - min;
            return Math.Max(upper - lower - (int)diff + 1, 0);
        }

        public int NumRabbits(int[] answers)
        {
            var rabbitsCount = 0;
            var dict = new Dictionary<int, int>();
            foreach (var r in answers)
            {
                if (r == 0)
                {
                    rabbitsCount++;
                    continue;
                }

                dict.TryAdd(r, 0);
                dict[r]++;
                if (dict[r] != r + 1)
                    continue;
                rabbitsCount += r + 1;
                dict.Remove(r);
            }

            foreach (var x in dict.Keys)
                rabbitsCount += dict[x] + 1;
            return rabbitsCount;
        }

        public long CountFairPairs(int[] nums, int lower, int upper)
        {
            Array.Sort(nums);
            return CountLess(nums, upper + 1) - CountLess(nums, lower);
        }

        long CountLess(int[] nums, int sum)
        {
            var count = 0L;
            var left = 0;
            var right = nums.Length - 1;
            while (left < right)
            {
                if (nums[left] + nums[right] < sum)
                {
                    count += right - left;
                    left++;
                }
                else
                    right--;
            }

            return count;
        }

        List<int> inorder = new();

        public IList<int> InorderTraversal(TreeNode root)
        {
            inorder = new List<int>();
            InorderTraverse(root);
            return inorder;
        }

        void InorderTraverse(TreeNode root)
        {
            if (root == null)
                return;
            InorderTraverse(root.left);
            InorderTraverse(root.right);
            inorder.Add(root.val);
        }

        public static string CountAndSay(int n)
        {
            var sb = new StringBuilder("1");
            for (var i = 0; i < n - 1; i++)
            {
                var count = 0;
                var current = sb[0];
                var newResult = new StringBuilder(2 * n);
                for (var j = 0; j < sb.Length; j++)
                {
                    if (sb[j] == current)
                    {
                        count++;
                    }
                    else
                    {
                        newResult.Append(count);
                        newResult.Append(current);
                        count = 1;
                        current = sb[j];
                    }

                    if (j == sb.Length - 1)
                    {
                        newResult.Append(count);
                        newResult.Append(current);
                    }
                }

                sb = new StringBuilder(newResult.ToString());
            }

            return sb.ToString();
        }

        public int CountPairs(int[] nums, int k)
        {
            var result = 0;
            for (var i = 0; i < nums.Length; i++)
            for (var j = i + 1; j < nums.Length; j++)
            {
                if (nums[i] != nums[j] || i * j % k != 0)
                    continue;
                result++;
            }

            return result;
        }

        public static long CountGood(int[] nums, int k)
        {
            var dict = new Dictionary<int, int>(nums.Length);
            var left = 0;
            var count = 0;
            long result = 0;

            for (var right = 0; right < nums.Length; right++)
            {
                var x = nums[right];
                dict.TryAdd(x, 0);
                var n = dict[x];
                count -= n * (n - 1) / 2;
                dict[x]++;
                n = dict[x];
                count += n * (n - 1) / 2;
                if (count < k)
                    continue;
                result += nums.Length - right;
                while (left < right && count >= k)
                {
                    var l = nums[left];
                    n = dict[l];
                    count -= n * (n - 1) / 2;
                    dict[l]--;
                    n = dict[l];
                    count += n * (n - 1) / 2;
                    left++;
                    if (count < k)
                        break;
                    result += nums.Length - right;
                }
            }

            return result;
        }

        public long GoodTriplets(int[] nums1, int[] nums2)
        {
            var n = nums1.Length;
            var pos2 = new int[n];
            var reversedIndexMapping = new int[n];
            for (var i = 0; i < n; i++)
            {
                pos2[nums2[i]] = i;
            }

            for (var i = 0; i < n; i++)
            {
                reversedIndexMapping[pos2[nums1[i]]] = i;
            }

            var tree = new FenwickTree(n);
            long res = 0;
            for (var i = 0; i < n; i++)
            {
                var pos = reversedIndexMapping[i];
                var left = tree.Query(pos);
                tree.Update(pos, 1);
                var right = n - 1 - pos - (i - left);
                res += (long)left * right;
            }

            return res;
        }
    }
}