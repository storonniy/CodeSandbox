namespace CodeSandbox.Leetcode.Medium;

public class ValidSudoku
{
    public bool IsValidSudoku(char[][] board)
    {
        var verticalByX = new Dictionary<int, HashSet<char>>();
        var horizontalByY = new Dictionary<int, HashSet<char>>();
        var quadByPoint = new Dictionary<(int X, int Y), HashSet<char>>();
        for (var x = 0; x < 9; x++)
        {
            for (var y = 0; y < 9; y++)
            {
                if (board[x][y] == '.')
                    continue;
                if (!verticalByX.ContainsKey(x))
                    verticalByX.Add(x, new HashSet<char>());
                if (!horizontalByY.ContainsKey(y))
                    horizontalByY.Add(y, new HashSet<char>());
                if (!quadByPoint.ContainsKey((x / 3, y / 3)))
                    quadByPoint.Add((x / 3, y / 3), new HashSet<char>());
                if (verticalByX[x].Contains(board[x][y]))
                    return false;
                if (horizontalByY[y].Contains(board[x][y]))
                    return false;
                if (quadByPoint[(x / 3, y / 3)].Contains(board[x][y]))
                    return false;
                verticalByX[x].Add(board[x][y]);
                horizontalByY[y].Add(board[x][y]);
                quadByPoint[(x / 3, y / 3)].Add(board[x][y]);
            }
        }

        return true;
    }
}