import axios from 'axios';
import { mutationLocalType } from './mutation-types';
import { actionLocalType } from './action-types';

const mutations = {
  [mutationLocalType.reset](state) {
    state.models = [];
    state.selectedModel = null;
  },

  [mutationLocalType.models](state, models) {
    state.models = [...models];
  },

  [mutationLocalType.setModel](state, model) {
    state.selectedModel = model;
  },
};

const actions = {
  [actionLocalType.getModels](context) {
    axios
      .get('/predict/')
      .then((response) => {
        context.commit(mutationLocalType.models, response.data);
      })
      .catch((error) => {
        console.error(error);
      });
  },
};

const getters = {};

const state = {
  models: [],
  selectedModel: null,
};

const mmsManagementModule = {
  namespaced: true,
  mutations,
  actions,
  getters,
  state,
};

export default mmsManagementModule;
