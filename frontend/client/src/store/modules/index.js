// Register each directory as a corresponding Vuex module. Modules are namespaced
// as the camelCase equivalent of their file name.

import camelCase from 'lodash/camelCase';

const modulesCache = {};
const storeData = { modules: {} };

// Recursively get the namespace of a Vuex module, even if nested
const getNamespace = function getNamespace(subtree, path) {
  if (path.length === 1) {
    return subtree;
  }

  const namespace = path.shift();
  // eslint-disable-next-line no-param-reassign
  subtree.modules[namespace] = {
    modules: {},
    namespaced: true,
    ...subtree.modules[namespace],
  };

  return getNamespace(subtree.modules[namespace], path);
};

const updateModules = function updateModules() {
  // Dynamically require all Vuex module
  // https://webpack.js.org/guides/dependency-management/#require-context
  const requireModule = require.context(
    // Search directory
    '.',
    // Search subdirectories
    true,
    // Search pattern
    /^\.\/\w+\/index.js$/,
  );

  // For every Vuex module
  requireModule.keys().forEach((fileName) => {
    const moduleDefinition = requireModule(fileName).default || requireModule(fileName);

    // Skip the module during hot reload if it refers to the
    // same module definition as the one we have cached
    if (modulesCache[fileName] === moduleDefinition) {
      return;
    }

    // Update the module cache, for efficient hot reloading
    modulesCache[fileName] = moduleDefinition;

    // Get the module path as an array
    const modulePath = fileName
      // Remove the "./" from the beginning
      .replace(/^\.\//, '')
      // Remove the file extension from the end
      .replace(/\.\w+$/, '')
      // Split nested modules into an array path
      .split(/\//)
      // camelCase all module namespaces and names
      .map(camelCase);

    // Remove "index"
    modulePath.pop();

    // Get the modules object for the current path
    const { modules } = getNamespace(storeData, modulePath);

    // Add the module to our modules object
    modules[modulePath.pop()] = {
      namespaced: true,
      ...moduleDefinition,
    };
  });

  // If the environment supports hot reloading
  if (module.hot) {
    // Whenever any Vuex module is updated
    module.hot.accept(requireModule.id, () => {
      // Update `storeData.modules` with the latest definitions
      updateModules();
      // Trigger a hot update in the store
      // eslint-disable-next-line global-require
      require('../index').default.hotUpdate({ modules: storeData.modules });
    });
  }
};

updateModules();

export default storeData.modules;
