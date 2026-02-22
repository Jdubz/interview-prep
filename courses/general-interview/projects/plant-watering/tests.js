const { test, assert, isSameTime } = require('./test-runner');
const service = require('./plantService');

// Note: There is no automatic cleanup of data between tests

const now = new Date('2026-01-10T00:00:00Z');

test('adding the same plant twice is idempotent', () => {
  const userId = 'user-823975';

  service.addPlant(userId, 'fern', 7);
  service.addPlant(userId, 'fern', 7);

  const plants = service.listPlants(userId, now);

  assert(plants.length === 1);
  assert(plants[0].plantId === 'fern');
  assert(plants[0].wateringIntervalDays === 7);
  assert(plants[0].lastWateredAt === null);
  assert(plants[0].needsWatering === true);
});

test("a user's plants are kept separate from those of other users", () => {
  const userId1 = 'user-329857';
  const userId2 = 'user-561901';

  service.addPlant(userId1, 'fern', 7);
  service.addPlant(userId2, 'fern', 5);

  const plants1 = service.listPlants(userId1, now);
  const plants2 = service.listPlants(userId2, now);

  assert(plants1.length === 1);
  assert(plants1[0].plantId === 'fern');
  assert(plants1[0].wateringIntervalDays === 7);

  assert(plants2.length === 1);
  assert(plants2[0].plantId === 'fern');
  assert(plants2[0].wateringIntervalDays === 5);
});

test('watering a plant updates it appropriately', () => {
  const userId = 'user-283957';

  service.addPlant(userId, 'cactus', 14);

  const wateredAt = new Date('2026-01-01T10:00:00Z');
  service.waterPlant(userId, 'cactus', wateredAt);

  const [plant] = service.listPlants(userId, now);
  const expectedDue = new Date('2026-01-15T10:00:00Z');

  assert(isSameTime(plant.lastWateredAt, wateredAt));
  assert(isSameTime(plant.nextWateringDueAt, expectedDue));
  assert(plant.needsWatering === false);
});

test('watering a non-existent plant is a no-op', () => {
  const userId = 'user-130914';

  service.waterPlant(userId, 'ghost-plant', new Date());

  const plants = service.listPlants(userId, now);
  assert(plants.length === 0);
});

test('a plant does not need watering before the interval has passed', () => {
  const userId = 'user-851650';

  service.addPlant(userId, 'ivy', 3);

  const wateredAt = new Date('2026-01-01T00:00:00Z');
  service.waterPlant(userId, 'ivy', wateredAt);

  const [plant] = service.listPlants(userId, new Date('2026-01-01T00:00:00Z'));

  assert(plant.needsWatering === false);
});

test('a plant needs watering once the interval has passed', () => {
  const userId = 'user-728356';

  service.addPlant(userId, 'ivy', 3);

  const wateredAt = new Date('2026-01-01T00:00:00Z');
  service.waterPlant(userId, 'ivy', wateredAt);

  const [plant] = service.listPlants(userId, now);

  assert(plant.needsWatering === true);
});

test('listing all of a users plants that new watering today', () => {
  const userId = 'user-999999';

  service.addPlant(userId, 'plant-1', 9);
  service.addPlant(userId, 'plant-2', 9);
  service.addPlant(userId, 'plant-3', 10);

  // test midnight cutover for needs watered today
  service.waterPlant(userId, 'plant-1', new Date('2026-01-01T00:00:00Z'));
  service.waterPlant(userId, 'plant-2', new Date('2026-01-02T23:00:00Z'));
  service.waterPlant(userId, 'plant-3', new Date('2026-01-01T00:00:00Z'));

  const plants = service.listPlants(userId, now);
  
  const needsWatering = plants.filter(plant => plant.needsWatering);

  assert(needsWatering.length === 2);
  const needsWateringIds = needsWatering.map(plant => plant.plantId);
  assert(needsWateringIds.includes('plant-2'));
});



console.log('\nAll tests completed');
