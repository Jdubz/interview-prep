
const db = new Map(); // userId -> Map(userId -> plantData)

/**
 * Adds a plant to be persisted in memory.
 */
function addPlant(
  userId, /* string */
  plantId, /* string */
  wateringIntervalDays /* integer */
) {
  
  let userPlants;
  if (!db.has(userId)) {
    userPlants = new Map();
    db.set(userId, userPlants);
  }
  userPlants = db.get(userId);

  userPlants.set(plantId, {
    plantId,
    wateringIntervalDays,
    lastWateredAt: null,
    nextWateringDueAt: null,
  });
}

/**
 * Returns array of plant objects, in no specific order.
 *
 * Each plant's shape should be:
 *   plantId              string
 *   wateringIntervalDays integer
 *   lastWateredAt        Date or null if never watered
 *   nextWateringDueAt    Date or null if never watered
 *   needsWatering        bool; true if due for watering as of `now` or never watered
 */
function listPlants(
  userId, /* string */
  now /* Date */
) {
  const userPlants = db.get(userId); 
  if (!userPlants) {
    return [];
  }
  const plantsWatered = Array.from(userPlants.values())
   .map(plant => {
    const needsWatering = plant.nextWateringDueAt <= now;
    return {
      ...plant,
      needsWatering,
    };
  }
   )
  return plantsWatered;
}

/**
 * Updates a plant, recording that it was watered at the specified time.
 */
function waterPlant(
  userId, /* string */
  plantId, /* string */
  wateredAt /* Date */
) {
  const userPlants = db.get(userId);
  if (!userPlants) {
    return;
  }
  const plant = userPlants.get(plantId);
  if (!plant) {
    return;
  }
  const nextWateringDueAt = new Date(wateredAt);
  nextWateringDueAt.setDate(
    nextWateringDueAt.getDate() + plant.wateringIntervalDays
  );
  plant.lastWateredAt = wateredAt;
  plant.nextWateringDueAt = nextWateringDueAt;
}

module.exports = {
  addPlant,
  listPlants,
  waterPlant,
}
