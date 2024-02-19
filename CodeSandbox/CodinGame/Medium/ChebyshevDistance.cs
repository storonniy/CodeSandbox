namespace CodeSandbox.CodinGame.Medium;

public class ChebyshevDistance
{


    private static void Meow(int[] collection)
    {
        for (var i = 0; i < collection.Length; i++)
            collection[i] = 3;
    }

    static void Run()
    {
        var inputs = Console.ReadLine()!.Split(' ')
            .Select(int.Parse)
            .ToArray();
        var radius = inputs[0];
        var x = inputs[1];
        var y = inputs[2];
        var vx = inputs[3];
        var vy = inputs[4];
        var time = inputs[5];
        Console.Error.WriteLine(String.Join(" ", inputs));
        var spaceship = new Spaceship(x, y, vx, vy);
        var visited = new HashSet<(int x, int y)>();
        var cycleStarted = false;
        var cycleFinished = false;
        var cycleCount = 0;
        var cycleLength = 0;
        var trajectory = new List<(int, int)>();
        for (var t = 0; t < time; t++)
        {
            spaceship.Move();
            if (!cycleStarted && visited.Contains((spaceship.X, spaceship.Y)))
            {
                cycleStarted = true;
                visited.Clear();
                Console.WriteLine($"cycle");
            }

            if (cycleStarted && cycleCount < cycleLength)
            {
                trajectory.Add((spaceship.X, spaceship.Y));
                cycleLength++;
            }

            if (cycleStarted && cycleCount == cycleLength)
            {
                cycleCount = 0;
            }

            if (cycleStarted && cycleCount >= cycleLength)
            {
                if (trajectory[cycleCount] != (spaceship.X, spaceship.Y))
                {
                    Console.WriteLine("Its not a cycle. Fuck");
                    cycleFinished = false;
                    cycleStarted = false;
                    cycleCount = 0;
                    cycleLength = 0;
                }
            }

            if (cycleStarted && !cycleFinished && visited.Contains((spaceship.X, spaceship.Y)))
            {
                cycleFinished = true;
                cycleLength = visited.Count;
                time = t + (time - t) % cycleLength;
                Console.WriteLine($"cycleFinished");
            }

            visited.Add((spaceship.X, spaceship.Y));
            if (spaceship.ChebyshevDistance <= radius)
            {
                Console.WriteLine($"{spaceship.X} {spaceship.Y} CRASH");
                return;
            }
        }

        Console.WriteLine($"{spaceship.X} {spaceship.Y} 0");
    }

    private class Spaceship
    {
        public int X { get; private set; }
        public int Y { get; private set; }
        int velocityX;
        int velocityY;
        public int dx { get; private set; }
        public int dy { get; private set; }
        public int ChebyshevDistance => Math.Max(Math.Abs(X), Math.Abs(Y));

        public Spaceship(int x, int y, int velocityX, int velocityY)
        {
            X = x;
            Y = y;
            this.velocityX = velocityX;
            this.velocityY = velocityY;
        }

        public void Move()
        {
            X += velocityX;
            Y += velocityY;
            ApplyGravity();
            Console.WriteLine($"{X} {Y}");
        }

        private void ApplyGravity()
        {
            if (X > 0)
                velocityX--;
            else if (X < 0)
                velocityX++;
            if (Y > 0)
                velocityY--;
            else if (Y < 0)
                velocityY++;
        }
    }
}