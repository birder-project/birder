import axios from 'axios';
import { mutationLocalType } from './mutation-types';
import { actionLocalType } from './action-types';

const mutations = {
  [mutationLocalType.reset](state) {
    state.languages = {};
    state.selectedLanguage = null;
  },

  [mutationLocalType.languages](state, languages) {
    state.languages = { ...languages };
  },

  [mutationLocalType.setLanguage](state, language) {
    state.selectedLanguage = language;
  },
};

const actions = {
  [actionLocalType.getLanguages](context) {
    axios
      .get('/language/')
      .then((response) => {
        context.commit(mutationLocalType.languages, response.data);
      })
      .catch((error) => {
        console.error(error);
      });
  },
};

const getters = {};

const state = {
  languages: {},
  selectedLanguage: null,
};

const languageModule = {
  namespaced: true,
  mutations,
  actions,
  getters,
  state,
};

export default languageModule;
