/* eslint-disable import/prefer-default-export */

const appendPrefix = function appendPrefix(moduleName, localType) {
  const types = {};
  Object.keys(localType).forEach((key) => {
    types[key] = `${moduleName}/${localType[key]}`;
  });

  return types;
};

export { appendPrefix };
