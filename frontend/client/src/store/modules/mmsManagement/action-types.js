import { appendPrefix } from '../../types';

const MODULE_NAME = 'mmsManagement';
const actionLocalType = {
  getModels: 'get_models',
};

const actionType = appendPrefix(MODULE_NAME, actionLocalType);

export { MODULE_NAME, actionLocalType };
export default actionType;
