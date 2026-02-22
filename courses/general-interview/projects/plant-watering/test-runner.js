function test(description, fn) {
  console.log(`\n→ ${description}`);
  try {
    fn();
    console.log('  ✅ passed');
  } catch (err) {
    console.error('  ❌ failed\n');
    console.error(`${err.stack}\n`);
    if (process.argv.includes('--fail-fast')) {
      process.exit(1);
    } else {
      process.exitCode = 1;
    }
  }
}

function assert(condition) {
  if (!condition) {
    throw new Error('Assertion failed');
  }
}

function isSameTime(a, b) {
  return new Date(a).getTime() === new Date(b).getTime();
}

module.exports = {
  test,
  assert,
  isSameTime,
};
