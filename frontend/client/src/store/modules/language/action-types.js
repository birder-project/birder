import { appendPrefix } from '../../types';

const MODULE_NAME = 'language';
const actionLocalType = {
  getLanguages: 'get_languages',
};

const actionType = appendPrefix(MODULE_NAME, actionLocalType);

export { MODULE_NAME, actionLocalType };
export default actionType;
