# Boulder Care Plant Watering Interview Exercise

Your first task is to write a barebones "service" (just a few functions) in vanilla JS to act as a
record keeping system for the watering schedule of house plants.

This exercise does not involve complex algorithms or obscure trivia. Our intent is to see how you
write basic, real-world-ish code, with boilerplate and libraries out of the way.

## Files

* `plantService.js`: A shell of an implementation. Implement the 3 functions here to satisfy the
  tests. Data can just be stored in memory and doesn't need to be persisted.
* `tests.js`: Pre-written tests. We want to make these pass.
* `test-runner.js`: Some vanilla JS helper functions to make the tests more readable and usable.

## Testing

Run `node tests.js` to run the tests. Failures will include stack traces.

Add `--fail-fast` to stop after the first failure.

## Bonus Tasks

If you have time left, pick and choose from these in any order:

* Improve input validation
  * We want to improve the robustness of the application, and prevent it from crashing or doing
    strange things when users input surprising values.
  * If you already focused a lot on validation, maybe skip this one.
* Support listing plants that need to be watered today
  * Not _now_, but any time today.
  * Don't let timezones bog you down! Do everything in UTC.
* Support updating the watering interval for a plant
