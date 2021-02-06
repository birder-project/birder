import Vue from 'vue';
import VueRouter from 'vue-router';

const DEFAULT_TITLE = 'Birder';

Vue.use(VueRouter);

const routes = [
  {
    path: '/',
    name: 'Home',
    component: () => import(/* webpackChunkName: "main" */ '../views/Home.vue'),
  },
  {
    path: '/about',
    name: 'About',
    component: () => import(/* webpackChunkName: "about" */ '../views/About.vue'),
  },
  {
    path: '*',
    component: () => import(/* webpackChunkName: "main" */ '../views/NotFound.vue'),
    meta: { title: `Not Found (404) - ${DEFAULT_TITLE}` },
    pathToRegexpOptions: { strict: true },
  },
];

const router = new VueRouter({
  mode: 'history',
  base: process.env.BASE_URL,
  routes,
});

export default router;
