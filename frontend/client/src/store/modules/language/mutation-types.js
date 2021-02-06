import { appendPrefix } from '../../types';

const MODULE_NAME = 'language';
const mutationLocalType = {
  languages: 'languages',
  setLanguage: 'set_language',
};

const mutationType = appendPrefix(MODULE_NAME, mutationLocalType);

export { MODULE_NAME, mutationLocalType };
export default mutationType;
