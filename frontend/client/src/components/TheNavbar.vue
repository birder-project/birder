<template>
  <nav class="navbar" role="navigation" aria-label="main navigation">
    <!-- Brand and burger bar -->
    <div class="navbar-brand">
      <RouterLink to="/" class="navbar-item">
        <img alt="Birder logo" src="@/assets/logo.png" />
      </RouterLink>

      <a
        role="button"
        class="navbar-burger burger"
        aria-label="menu"
        aria-expanded="false"
        v-on:click="navBurgerActive = !navBurgerActive"
        v-bind:class="{ 'is-active': navBurgerActive }"
      >
        <span aria-hidden="true"></span>
        <span aria-hidden="true"></span>
        <span aria-hidden="true"></span>
      </a>
    </div>

    <!-- Menu -->
    <div class="navbar-menu" v-bind:class="{ 'is-active': navBurgerActive }">
      <div class="navbar-start">
        <!-- Navigation links -->
        <RouterLink to="/" class="navbar-item">Home</RouterLink>
        <RouterLink to="/about/" class="navbar-item">About</RouterLink>
      </div>
    </div>
  </nav>
</template>

<script>
export default {
  components: {},

  data() {
    return {
      navBurgerActive: false,
    };
  },
};
</script>
