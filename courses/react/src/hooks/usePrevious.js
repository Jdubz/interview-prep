import { useState } from 'react';

/**
 * EXERCISE: Implement usePrevious
 *
 * This hook should return the previous value of whatever is passed to it.
 *
 * Example usage:
 *   const [count, setCount] = useState(0);
 *   const prevCount = usePrevious(count);
 *   // First render: prevCount = undefined
 *   // After setCount(1): prevCount = 0
 *   // After setCount(5): prevCount = 1
 *
 * Hint: Think about when useRef updates vs when useEffect runs
 */

export function usePrevious(value) {
  // TODO: Implement this hook
  //
  // Questions to consider:
  // 1. Where should you store the previous value?
  // 2. When should you update the stored value?
  // 3. What should you return?
  const [[prev, curr], setState] = useState([undefined, value]);

  if(curr !== value) {
    setState([curr, value])
  }

  return prev;
}
