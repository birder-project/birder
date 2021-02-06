import { appendPrefix } from '../../types';

const MODULE_NAME = 'mmsManagement';
const mutationLocalType = {
  models: 'models',
  setModel: 'set_model',
};

const mutationType = appendPrefix(MODULE_NAME, mutationLocalType);

export { MODULE_NAME, mutationLocalType };
export default mutationType;
