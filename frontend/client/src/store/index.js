import Vue from 'vue';
import Vuex from 'vuex';
import modules from './modules';

Vue.use(Vuex);

export default new Vuex.Store({
  strict: process.env.NODE_ENV !== 'production', // Strict mode only in development / testing
  state: {},
  mutations: {},
  actions: {},
  modules,
});
