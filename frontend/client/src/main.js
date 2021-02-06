import Vue from 'vue';
import axios from 'axios';
import App from './App.vue';
import router from './router';
import store from './store';

axios.defaults.baseURL = `${process.env.VUE_APP_API_HOST}/inference/v1`;
axios.defaults.timeout = 2500;

Vue.config.productionTip = false;
Vue.config.devtools = false;

new Vue({
  router,
  store,
  render: (h) => h(App),
}).$mount('#app');
