namespace CodeSandbox.Leetcode.Medium;

public class DeciBinary
{
    //https://leetcode.com/problems/partitioning-into-minimum-number-of-deci-binary-numbers/
    /*
     * Максимальная цифра говорит о максимальном количестве полубинарных чисел, которые надо сложить, чтобы
     * получить искомое: максимальнае число, стоящее в любом разряде, - это 1
     */
    public int MinPartitions(string n) => n.Max() - '0';
}